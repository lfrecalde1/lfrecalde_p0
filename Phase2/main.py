# main.py
# ---------------------------------------------------------------------
# Aerial Robotics — Quadrotor Navigation Assignment (NED frame)
#
# ✍️ What YOU must implement:
#   1) load_waypoints_csv(csv_path): read (N,E,D) waypoints from CSV.
#      IMPORTANT: Waypoints are in the NED coordinate frame.
#        • N (North): meters, positive forward
#        • E (East):  meters, positive right
#        • D (Down):  meters, positive downward (altitude ↑ means D ↓)
#
#   2) Landing + Takeoff + Resume logic in guidance_update():
#      • Start landing once when D > 0.18 (i.e., you’re close to the ground),
#        then descend to a landing depth, take off (ascend), and RESUME
#        navigating toward the remaining/last waypoint.
#      • Ensure the drone lands AT MOST ONCE per run.
#
# ✅ What’s already provided:
#   • quadrotor rigid-body physics (quad_dynamics.py)
#   • a stabilizing controller (control.py: quad_control)
#   • compulsory rendering via a neural renderer (splat_render.py)
#   • plotting and video export (utils.py)
#
# 🧭 Tips for (2):
#   • Use a simple state machine: NAV → LANDING → TAKEOFF → NAV → … → DONE
#   • Track one-time landing with a boolean flag (e.g., has_landed_once).
#   • During LANDING: keep N,E fixed; increase D toward landing depth (e.g., 0.22).
#   • During TAKEOFF: keep N,E fixed; decrease D toward the next waypoint’s D
#     (or the last waypoint’s D if you’re near the end).
#   • You DO NOT implement low-level control — we give you controller_step().
# ---------------------------------------------------------------------

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless-friendly
import matplotlib

matplotlib.use("Agg")

import csv
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import numpy as np
import scipy.io
import cv2
import importlib
from pyquaternion import Quaternion

# --- project modules ---
import quad_dynamics as qd
import ctrlmod as control
import drone
import utils
from utils import Config, Paths

import matplotlib.pyplot as plt
import scienceplots

from generate_waypoints import (
    generate_3d_infinity_trajectory,
    sample_trajectory_points,
    save_trajectory_plots,
)

# ---------- Mandatory renderer ----------
from splat_render import SplatRenderer

HAS_SPLAT = True
print(HAS_SPLAT)


# =================== High-level guidance states ===================
class Mode(str, Enum):
    NAV = "NAV"  # normal waypoint navigation
    LANDING = "LANDING"  # controlled descend to landing depth
    TAKEOFF = "TAKEOFF"  # controlled ascend after landing
    DONE = "DONE"  # finished (all waypoints reached)


@dataclass
class AppState:
    """Runtime variables that the main loop mutates."""

    t: float
    X: np.ndarray  # full state vector
    mode: Mode
    wp_idx: int
    wps_done: bool
    has_landed_once: bool
    landing_depth: Optional[float]
    landing_start_t: Optional[float]
    landing_end_t: Optional[float]
    takeoff_target_D: Optional[float]
    control_cd: float
    user_cd: float
    render_cd: float
    render_every: float
    render_idx: int
    # (Optional) If you want to visualize takeoff separately later:
    takeoff_start_t: Optional[float] = None
    takeoff_end_t: Optional[float] = None


# =================== Utility functions (provided) ===================
def ensure_dirs(paths: Paths) -> None:
    for d in [paths.LOG_DIR, paths.RENDER_DIR, paths.VIDEO_DIR, paths.PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def quat_to_rpy_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    """q=[w,x,y,z] -> [roll, pitch, yaw] (radians)."""
    qw, qx, qy, qz = q_xyzw
    q = Quaternion(qw, qx, qy, qz)
    yaw, pitch, roll = q.yaw_pitch_roll
    return np.array([roll, pitch, yaw], dtype=np.float32)


def init_renderer(cfg: Config):
    """Rendering is COMPULSORY. If unavailable, fail fast with a clear error."""
    if not HAS_SPLAT:
        raise RuntimeError(
            "SplatRenderer not available — rendering is compulsory for this assignment."
        )
    return SplatRenderer(cfg.render.config_path, cfg.render.json_path)


def save_render_frame(
    paths: Paths, rgb: np.ndarray, depth: np.ndarray, idx: int
) -> None:
    rgb_path = os.path.join(paths.RENDER_DIR, f"rgb_{idx:05d}.jpg")
    depth_path = os.path.join(paths.RENDER_DIR, f"depth_{idx:05d}.jpg")
    depth_gray = (
        cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        if (depth.ndim == 3 and depth.shape[2] == 3)
        else depth
    )
    cv2.imwrite(rgb_path, rgb)
    cv2.imwrite(depth_path, depth_gray)


def init_state(start_pos_ned: np.ndarray, cam_pitch_deg: float) -> np.ndarray:
    """Build initial state vector X from start position & camera pitch (NED)."""
    xyz = start_pos_ned.copy()  # [N,E,D]
    vxyz = np.zeros(3)  # [vN,vE,vD]
    quat = np.array(
        [  # unit quaternion [w,x,y,z]
            np.cos(np.deg2rad(cam_pitch_deg) / 2.0),
            0.0,
            np.sin(np.deg2rad(cam_pitch_deg) / 2.0),
            0.0,
        ]
    )
    pqr = np.zeros(3)  # body rates [p,q,r]
    # State vector
    # Position
    # Velocity
    # Quad
    # Angular velocity
    return np.concatenate([xyz, vxyz, quat, pqr])


# TODO - Modify the thresholds to get your drone till the end of the trajectory #######
def get_next_waypoint(
    waypoints: np.ndarray,
    current_idx: int,
    current_pos: np.ndarray,
    threshold: float = 0.15,
):
    """
    Simple waypoint sequencer: if we're within 'threshold' meters of the current waypoint,
    advance to the next one.
    Returns: (wp, new_idx, finished_flag)
    """
    if current_idx >= len(waypoints):
        return waypoints[-1], current_idx, True
    wp = waypoints[current_idx]
    if np.linalg.norm(current_pos - wp) < threshold:
        current_idx += 1
        print(f"Reached waypoint {current_idx}/{len(waypoints)}")
        if current_idx >= len(waypoints):
            return waypoints[-1], current_idx, True
        wp = waypoints[current_idx]
    return wp, current_idx, False


def controller_step(ctrl, X, WP, V, A) -> np.ndarray:
    """
    Low-level control hookup (DO NOT MODIFY for this assignment).
    Given:
      X  : full state
      WP : [N_sp, E_sp, D_sp, yaw_sp]
      V  : desired linear velocity [vN, vE, vD] (optional, can be zeros)
      A  : desired linear acceleration [aN, aE, aD] (optional, can be zeros)
    Returns:
      U: the control input for the dynamics integrator.
    """
    return ctrl.step(X, WP, V, A)


def integrate_dynamics(
    cfg: Config, t: float, X: np.ndarray, U: np.ndarray
) -> np.ndarray:
    """Semi-implicit Euler step for the rigid-body model."""
    return X + cfg.sim.dynamics_dt * qd.model_derivative(t, X, U, drone)


# =================== PART (A): Waypoint CSV loader ===================
def load_waypoints_csv(csv_path: str) -> np.ndarray:
    """
    Load waypoints from a CSV file with columns N, E, D (North, East, Down).

    - Handles both with/without header.
    - Skips bad or empty rows.
    - Returns np.ndarray of shape [K, 3], dtype float32.
    """
    waypoints = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            # Skip empty rows
            if not row:
                continue

            try:
                n, e, d = map(float, row[:3])  # take first 3 columns
                waypoints.append([n, e, d])
            except ValueError:
                # If it's the first row, probably a header → skip
                if i == 0:
                    continue
                # Otherwise, just ignore malformed rows
                else:
                    continue

    return np.array(waypoints, dtype=np.float32)


def build_waypoints(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the list of navigation waypoints.

    Behavior:
      • If cfg.traj.use_csv and the file exists, use the CSV (students' loader).
      • Otherwise, generate a default 3D "infinity" trajectory and sample points.

    Returns:
      sampled_wps: [K,3]  — the navigation waypoints (N,E,D).
      full_traj:   [N,3]  — dense param. trajectory (for plotting context).
      times:       [N]    — param. times for the dense trajectory.
    """
    if cfg.traj.use_csv and os.path.exists(cfg.traj.csv_path):
        sampled = load_waypoints_csv(cfg.traj.csv_path)  # ← your function
        full_traj = sampled.copy()
        times = np.linspace(0.0, cfg.traj.total_time, len(full_traj), dtype=np.float32)
        print(f"[waypoints] Loaded {len(sampled)} from {cfg.traj.csv_path}")
        return sampled, full_traj, times

    # Fallback: generate an infinity trajectory and sample K waypoints.
    full_traj, times = generate_3d_infinity_trajectory(
        total_time=cfg.traj.total_time,
        dt=0.05,
        z_high=cfg.traj.z_high,
        z_low=cfg.traj.z_low,
        radius_x=cfg.traj.radius_x,
        radius_y=cfg.traj.radius_y,
        center=(0.0, 0.0),
    )
    sampled, _, _ = sample_trajectory_points(
        full_traj,
        times,
        num_samples=cfg.traj.num_waypoints,
        ordered=not cfg.traj.randomize_waypoints,
        random_seed=None,
    )
    save_trajectory_plots(
        full_traj,
        times,
        sampled_points=sampled,
        z_high=cfg.traj.z_high,
        z_low=cfg.traj.z_low,
        save_dir=cfg.paths.PLOTS_DIR,
    )
    print(
        f"[waypoints] Generated {len(sampled)} sampled points from infinity trajectory"
    )
    return sampled, full_traj, times


# =================== PART (B): Landing/Takeoff ===================
def guidance_update(
    state: AppState,
    cfg: Config,
    sampled_wps: np.ndarray,
    WP: np.ndarray,
    V: np.ndarray,
    A: np.ndarray,
    gt_wp_log: List[np.ndarray],
    exec_wp_log: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Update guidance setpoints based on the current mode (state machine).

    Implements:
      - One-time landing trigger in NAV when D > 0.18
      - LANDING: go down to final landing depth (e.g., 0.22)
      - TAKEOFF: go back up to the next waypoint's D, then resume NAV
    """
    # Tunables
    LAND_TRIGGER_D = 0.18  # start landing if below ~18 cm altitude (D positive down)
    FINAL_LAND_D = cfg.lt.land_final_depth
    LAND_STEP = 0.001  # per-tick D increment (down)
    TAKEOFF_STEP = 0.001  # per-tick D decrement (up)
    D_EPS = 0.01  # tolerance for reaching targets
    V_D_DOWN = 0.01  # optional desired vertical speed down (+D)
    V_D_UP = -0.005  # optional desired vertical speed up (-D)

    pos_NED = state.X[0:3].copy()
    N_now, E_now, D_now = pos_NED

    # ---------------- NAV mode ----------------
    if state.mode == Mode.NAV:
        # One-time landing trigger
        if (
            (not state.wps_done)
            and (not state.has_landed_once)
            and (state.t > 2.0)
            and (D_now > LAND_TRIGGER_D)
        ):
            state.mode = Mode.LANDING
            state.landing_depth = D_now
            state.landing_start_t = state.t
            state.render_every = cfg.render.every_sec_land
            print(f"[LANDING] start t={state.t:.2f}s, D={D_now:.3f}")

        # Normal waypoint following while still in NAV
        if state.mode == Mode.NAV:
            target_wp, state.wp_idx, state.wps_done = get_next_waypoint(
                sampled_wps, state.wp_idx, pos_NED
            )
            if state.wps_done:
                print(f"[NAV] all waypoints completed t={state.t:.2f}s")
                state.mode = Mode.DONE
                target_wp = pos_NED  # hold
            n_sp, e_sp, d_sp = target_wp
            yaw_sp = 0.0
            WP[:] = np.array([n_sp, e_sp, d_sp, yaw_sp], dtype=np.float32)
            V[:] = 0.0
            A[:] = 0.0
            state.render_every = cfg.render.every_sec_nav
            gt_wp_log.append(target_wp.copy())
            exec_wp_log.append(WP[:3].copy())

    # ---------------- LANDING mode ----------------
    elif state.mode == Mode.LANDING:
        # Hold current N,E; increase D toward FINAL_LAND_D
        d_target = min(D_now + LAND_STEP, FINAL_LAND_D)
        WP[:] = np.array([N_now, E_now, d_target, 0.0], dtype=np.float32)

        # Optional vertical velocity/accel hints
        V[:] = 0.0
        V[2] = 0.0
        A[:] = 0.0

        # Log targets
        last_idx = max(0, min(state.wp_idx, len(sampled_wps) - 1))
        gt_wp_log.append(sampled_wps[last_idx].copy())
        exec_wp_log.append(WP[:3].copy())

        # Switch to TAKEOFF when close to ground
        if abs(FINAL_LAND_D - d_target) <= D_EPS:
            state.landing_end_t = state.t
            state.has_landed_once = True

            # Decide where to take off to: next pending waypoint's D (or last WP’s D)
            if state.wp_idx < len(sampled_wps):
                next_idx = state.wp_idx  # resume current/next pending
            else:
                next_idx = len(sampled_wps) - 1
            state.takeoff_target_D = float(sampled_wps[max(0, next_idx)][2])

            state.mode = Mode.TAKEOFF
            state.render_every = cfg.render.every_sec_nav
            print(f"[TAKEOFF] target D={state.takeoff_target_D:.3f} t={state.t:.2f}s")

    # ---------------- TAKEOFF mode ----------------
    elif state.mode == Mode.TAKEOFF:
        # Hold current N,E; decrease D toward takeoff target
        tgtD = getattr(state, "takeoff_target_D", D_now)
        # move upward (decrease D)
        stepD = max(tgtD, D_now - TAKEOFF_STEP)
        WP[:] = np.array([N_now, E_now, stepD, 0.0], dtype=np.float32)

        V[:] = 0.0
        V[2] = 0.0
        A[:] = 0.0

        # Log
        resume_idx = max(0, min(state.wp_idx, len(sampled_wps) - 1))
        gt_wp_log.append(sampled_wps[resume_idx].copy())
        exec_wp_log.append(WP[:3].copy())

        # Reached climb target? return to NAV
        if abs(stepD - tgtD) <= D_EPS:
            state.mode = Mode.NAV
            # (optional) state.takeoff_end_t = state.t
            print(f"[NAV] resume t={state.t:.2f}s, D≈{tgtD:.3f}")

    # ---------------- DONE mode ----------------
    elif state.mode == Mode.DONE:
        # Hold current pose
        WP[:] = np.array([state.X[0], state.X[1], state.X[2], 0.0], dtype=np.float32)
        V[:] = 0.0
        A[:] = 0.0
        gt_wp_log.append(state.X[0:3].copy())
        exec_wp_log.append(WP[:3].copy())

    return WP, V, A


# =================== Rendering + Logging (provided) ===================
def render_if_needed(
    state: AppState, cfg: Config, renderer, render_times: List[float]
) -> None:
    """Render the RGB/Depth frame at the current pose on a fixed cadence."""
    if state.render_cd > 0.0:
        return
    ex, ey, ez = state.X[0], state.X[1], state.X[2]
    qw, qx, qy, qz = state.X[6:10]
    rpy = quat_to_rpy_xyzw(np.array([qw, qx, qy, qz]))
    pose = np.array([ex, ey, ez])  # NED position
    rgb, depth = renderer.render(pose, rpy)  # may raise if renderer misconfigured
    save_render_frame(cfg.paths, rgb, depth, state.render_idx)
    render_times.append(state.t)
    state.render_idx += 1
    state.render_cd = state.render_every


def save_logs(
    cfg: Config,
    t_log,
    X_log,
    U_log,
    gt_wp_log,
    exec_wp_log,
    sampled_wps,
    full_traj,
    render_times,
    landing_mask,
):
    """Persist results to disk for later inspection (MAT + CSV)."""
    scipy.io.savemat(
        os.path.join(cfg.paths.LOG_DIR, "simulation_data.mat"),
        {
            "time": np.asarray(t_log),
            "state": np.asarray(X_log),
            "control": np.asarray(U_log),
            "gt_waypoints": np.asarray(gt_wp_log),
            "executed_waypoints": np.asarray(exec_wp_log),
            "sampled_waypoints": np.asarray(sampled_wps),
            "full_trajectory": np.asarray(full_traj),
            "render_times": np.asarray(render_times),
            "landing_mask": np.asarray(landing_mask),
        },
    )
    with open(
        os.path.join(cfg.paths.LOG_DIR, "gt_waypoints.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["N", "E", "D"])
        w.writerows(np.asarray(gt_wp_log).tolist())
    with open(
        os.path.join(cfg.paths.LOG_DIR, "executed_waypoints.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["N", "E", "D"])
        w.writerows(np.asarray(exec_wp_log).tolist())
    scipy.io.savemat(
        os.path.join(cfg.paths.LOG_DIR, "gt_vs_exec.mat"),
        {
            "time": np.asarray(t_log),
            "gt_pos_ned": np.asarray(gt_wp_log),
            "exec_pos_ned": np.asarray(exec_wp_log),
        },
    )


def post_plots_and_video(
    cfg: Config, USER_DT: float, t_log, X_log, gt_wp_log, landing_mask, render_times
):
    """Produce static plots and a 3‑panel RGB|Depth|3D video."""
    utils.save_separate_trajectory_plots(cfg.paths.LOG_DIR, USER_DT, PLOT_3D=True)
    utils.create_render_plot_video(
        render_dir=cfg.paths.RENDER_DIR,
        out_path=cfg.paths.COMBINED_VIDEO,
        fps=cfg.render.fps,
        t=np.asarray(t_log),
        pos_ned=np.asarray(X_log)[:, 0:3],
        gt_points=np.asarray(gt_wp_log) if len(gt_wp_log) else None,
        landing_mask=np.asarray(landing_mask),
        render_times=render_times,
        stop_at_landing=False,  # include takeoff + resume, if implemented
    )


# =================== Main entry (provided) ===================
def main():
    # Live-reload local modules (useful during iterative development).
    importlib.reload(control)
    importlib.reload(qd)
    importlib.reload(drone)

    # Load config and ensure output folders exist.
    cfg = Config()
    ensure_dirs(cfg.paths)

    # Create the controller and get its internal update rate.
    controller = control.quad_control()
    control_dt = controller.dt

    # Waypoints: either your CSV or the fallback infinity trajectory.
    sampled_wps, full_traj, traj_times = build_waypoints(cfg)
    ## Initialize the drone state: start at the first waypoint’s position with a small camera pitch.
    X0 = init_state(sampled_wps[0], cfg.sim.cam_pitch_deg)

    ## Initialize runtime state (starts in NAV mode).
    state = AppState(
        t=0.0,
        X=X0,
        mode=Mode.NAV,
        wp_idx=0,
        wps_done=False,
        has_landed_once=False,
        landing_depth=None,
        landing_start_t=None,
        landing_end_t=None,
        takeoff_target_D=None,
        control_cd=0.0,
        user_cd=0.0,
        render_cd=0.0,
        render_every=cfg.render.every_sec_nav,
        render_idx=0,
    )

    ## Renderer (compulsory). Fails immediately if not available.
    renderer = init_renderer(cfg)

    ## Logs
    t_log: List[float] = []
    X_log: List[np.ndarray] = []
    Xd_log: List[np.ndarray] = []
    U_log: List[np.ndarray] = []
    gt_wp_log: List[np.ndarray] = []
    exec_wp_log: List[np.ndarray] = []
    render_times: List[float] = []

    ## Guidance buffers (setpoints)
    WP = np.zeros(4, dtype=np.float32)  # [N,E,D,yaw]
    V = np.zeros(3, dtype=np.float32)  # desired linear velocity
    A = np.zeros(3, dtype=np.float32)  # desired linear acceleration

    sim_wall_start = time.time()

    ## ===== Main simulation loop =====
    while state.t < cfg.sim.stop_time and state.mode != Mode.DONE:
        # High-level guidance update (where you implement landing/takeoff logic)
        if state.user_cd <= 0.0:
            WP, V, A = guidance_update(
                state, cfg, sampled_wps, WP, V, A, gt_wp_log, exec_wp_log
            )
            state.user_cd = cfg.sim.user_dt

        # Low-level controller update (already implemented and available to you)
        if state.control_cd <= 0.0:
            U = controller_step(controller, state.X, WP, V, A)
            state.control_cd = control_dt

        #    # Integrate the rigid-body dynamics
        state.X = integrate_dynamics(cfg, state.t, state.X, U)

        #    # Render at a fixed cadence (saves RGB & Depth frames)
        render_if_needed(state, cfg, renderer, render_times)

        #    # Update loop timers
        state.control_cd -= cfg.sim.dynamics_dt
        state.user_cd -= cfg.sim.dynamics_dt
        state.render_cd -= cfg.sim.dynamics_dt
        state.t += cfg.sim.dynamics_dt

        #    # Log state & control
        t_log.append(state.t)
        X_log.append(state.X.copy())
        U_log.append(U.copy())
        Xd_log.append(WP)

    sim_wall_dur = time.time() - sim_wall_start
    print(f"Simulation wall time: {sim_wall_dur:.2f}s")

    ## Mark the landing interval for visual overlays (if you recorded it)
    t_arr = np.asarray(t_log)
    landing_mask = np.zeros_like(t_arr, dtype=bool)
    if (state.landing_start_t is not None) and (state.landing_end_t is not None):
        landing_mask = (t_arr >= state.landing_start_t) & (t_arr <= state.landing_end_t)

    ## Persist results and make plots & video
    save_logs(
        cfg,
        t_log,
        X_log,
        U_log,
        gt_wp_log,
        exec_wp_log,
        sampled_wps,
        full_traj,
        render_times,
        landing_mask,
    )
    post_plots_and_video(
        cfg, cfg.sim.user_dt, t_log, X_log, gt_wp_log, landing_mask, render_times
    )

    print("Done.")
    print(f"  Logs:   {cfg.paths.LOG_DIR}")
    print(f"  Plots:  {cfg.paths.PLOTS_DIR}")
    print(f"  Video:  {cfg.paths.COMBINED_VIDEO}")

    # after you build arrays:
    states = np.array(X_log)  # [N, 13]
    times = np.array(t_log)  # [N]
    states_d = np.array(Xd_log)  # [N, 13]

    print(states_d.shape)

    plot_z(times, states[:, 2], states_d[:, 2])


def plot_z(time, z, z_des):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(time, z, label="z current")
        ax.plot(time, z_des, "--", label="z desired")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("D [m]")  # NED: positive down
        ax.legend()
        ax.autoscale(tight=True)
        fig.savefig("z_vs_desired.pdf", dpi=300, bbox_inches="tight")
    return None


if __name__ == "__main__":
    main()
