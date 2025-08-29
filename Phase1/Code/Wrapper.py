import argparse
from pathlib import Path
import numpy as np
from scipy import io
from functions.plots_p0 import plot_samples, plot_acc, plot_gyro, plot_angles, plot_all_methods, plot_quaternions
from functions.video import make_orientation_video
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp
import casadi as ca

def _overlap_window(t1, t2):
    t0 = max(t1.min(), t2.min())
    t1_ = min(t1.max(), t2.max())
    if t1_ <= t0:
        raise ValueError("No time overlap between the two sequences.")
    return t0, t1_

def _median_dt(t):
    return float(np.median(np.diff(t)))

def _choose_target_time(t_imu, t_vicon, prefer="denser"):
    t0, t1 = _overlap_window(t_imu, t_vicon)

    imu_in = (t_imu >= t0) & (t_imu <= t1)
    vic_in = (t_vicon >= t0) & (t_vicon <= t1)

    if prefer == "imu":
        return t_imu[imu_in]
    if prefer == "vicon":
        return t_vicon[vic_in]

    # prefer == "denser"
    imu_dt = _median_dt(t_imu[imu_in])
    vic_dt = _median_dt(t_vicon[vic_in])
    # smaller dt = denser
    if imu_dt <= vic_dt:
        print("Imu TIME")
        print(imu_dt)
        return t_imu[imu_in]
    else:
        print("Vicon TIME")
        print(vic_dt)
        return t_vicon[vic_in]
def interp_linear_timeseries(t_src, X_src, t_tgt):
    t_src = np.asarray(t_src).ravel()
    t_tgt = np.asarray(t_tgt).ravel()

    if X_src.ndim == 1:
        return np.interp(t_tgt, t_src, X_src)

    D, N = X_src.shape
    X_out = np.empty((D, t_tgt.size), dtype=float)
    for i in range(D):
        X_out[i, :] = np.interp(t_tgt, t_src, X_src[i, :])
    return X_out

def slerp_rotmats(t_src, R_src_3x3xN, t_tgt):
    N = R_src_3x3xN.shape[2]
    # Build Rotation sequence
    Rs = R.from_matrix(np.transpose(R_src_3x3xN, (2, 0, 1)))  # (N,3,3)
    slerp = Slerp(np.asarray(t_src).ravel(), Rs)
    Rtgt = slerp(np.asarray(t_tgt).ravel())  # Rotation object at target times
    return Rtgt

def angles_from_rotation_obj_xyz(Robj):
    eul_zyx = Robj.as_euler('xyz', degrees=False)
    rpy = np.vstack([eul_zyx[:, 2], eul_zyx[:, 1], eul_zyx[:, 0]])
    return rpy

def quat_from_matrix(Rm):
    Rm = Rm.as_matrix()
    q = np.zeros((4, Rm.shape[0]))
    for k in range(0, Rm.shape[0]):
        r = R.from_matrix(Rm[k, :, :])
        x, y, z, w = r.as_quat()   # scipy returns (x, y, z, w)
        q[:, k] = np.array([w, x, y, z], dtype=float)
        # ensure unit norm (guards against tiny numerical drift)
        q[:, k] /= np.linalg.norm(q[:, k])

    # --- Fix double-cover discontinuities (sign flips) ---
    for k in range(1, Rm.shape[0]):
        if np.dot(q[:, k-1], q[:, k]) < 0.0:
            q[:, k] *= -1.0
    return q

def scale_measurements(imu, parameters):
    scales = parameters[0, :]
    biases = parameters[1, :]

    # Acc
    imu_filtered_empty = np.zeros_like(imu[0:3, :], dtype=float)

    # Gyro
    gyro_filtered_empy = np.zeros_like(imu[3:6, :], dtype=float)
    gyro_mean = np.mean(imu[3:6, 0:200], axis=1)
    
    # Filter Data For Loop
    for k in range(0, imu_filtered_empty.shape[1]):
        # Acc
        imu_filtered_empty[0, k] = ((imu[0, k]*scales[0]) + biases[0])*9.8
        imu_filtered_empty[1, k] = ((imu[1, k]*scales[1]) + biases[1])*9.8
        imu_filtered_empty[2, k] = ((imu[2, k]*scales[2]) + biases[2])*9.8

        # Gyro
        gyro_filtered_empy[:, k] = (3300.0/1023.0)*(np.pi/180.0)*(0.3)*(imu[3:6, k] - gyro_mean)

    return imu_filtered_empty, gyro_filtered_empy

def angles_from_acc(acc):
    rpy = np.zeros_like(acc)

    for k in range(0, rpy.shape[1]):
        rpy[2, k] = np.arctan((np.sqrt(acc[0, k]**2 + acc[1, k]**2))/acc[2, k])
        rpy[0, k] = np.arctan2(acc[1, k], acc[2, k])
        rpy[1, k] = -np.arctan2(acc[0, k], np.sqrt(acc[1, k]**2 + acc[2, k]**2))

    return rpy


def angles_from_rot(rot):
    M = rot.shape[2]
    rpy = np.zeros((3, M))
    for k in range(M):
        eul_zyx = R.from_matrix(rot[:, :, k]).as_euler('xyz', degrees=False)
        rpy[:, k] = np.array([eul_zyx[2], eul_zyx[1], eul_zyx[0]])
    return rpy

def euler_dot(euler, omega_imu):
    roll, pitch, yaw = euler.flatten()
    p_imu, q_imu, r_imu = omega_imu.flatten()

    # Map IMU gyro to body/Vicon frame
    p = p_imu
    q = q_imu
    r = r_imu

    sphi, cphi = np.sin(roll), np.cos(roll)
    tth, cth = np.tan(pitch), np.cos(pitch)

    # Transformation matrix from body rates -> Euler rates
    T = np.array([
        [1.0, sphi*tth, cphi*tth],
        [0.0, cphi,    -sphi],
        [0.0, sphi/cth, cphi/cth]
    ])

    omega_b = np.array([[p], [q], [r]])
    return T @ omega_b  # (3,1)

def integrate_gyro_euler(gyro, rpy0, t):
    M = gyro.shape[1]
    rpy = np.zeros((3, M))
    rpy[:, 0] = rpy0
    for k in range(M-1):
        dt = (t[k+1] - t[k])
        x  = rpy[:, k].reshape(3, 1)
        u = np.array([gyro[0, k], -gyro[1, k], -gyro[2, k]])
        u  = u.reshape(3, 1)

        k1 = euler_dot(x,            u)
        k2 = euler_dot(x + 0.5*dt*k1, u)
        k3 = euler_dot(x + 0.5*dt*k2, u)
        k4 = euler_dot(x + dt*k3,    u)

        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        rpy[:, k+1] = x_next[:, 0]
    return rpy


def low_passs_filter(signal, parameter):
    
    filter = np.zeros_like(signal)
    gain = np.array([[1.0 - parameter, 0.0, 0.0], [0.0, 1.0-parameter, 0.0],
                     [0.0, 0.0, 1.0-parameter]])
    I = np.array([[parameter, 0.0, 0.0], [0.0, parameter, 0.0],
                     [0.0, 0.0, parameter]])
    # Init Values
    filter[:, 0] = signal[:, 0]

    for k in range (0, signal.shape[1]-1):
        filter[:, k + 1] = gain@signal[:, k + 1] + I@filter[:, k]
    return filter

def high_pass_filter(x, fc, Ts):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = tau / (tau + Ts)
    y[:, 0] = 0.0
    for k in range(1, x.shape[1]):
        y[:, k] = alpha * (y[:, k-1] + x[:, k] - x[:, k-1])
    return y

def quat_to_euler_xyz(q):
    q_xyzw = np.column_stack([q[1, :], q[2, :], q[3, :], q[0, :]])
    r = R.from_quat(q_xyzw)
    eulers = r.as_euler('xyz', degrees=False)
    return eulers.T

def quatdot_np(q, omega, K_quat=0.0):
    q = np.asarray(q, dtype=float).reshape(4)
    w, x, y, z = q
    wx, wy, wz = np.asarray(omega, dtype=float).reshape(3)

    quat_err = 1.0 - (w*w + x*x + y*y + z*z)

    H_r_plus = np.array([
        [ w, -x, -y, -z],
        [ x,  w, -z,  y],
        [ y,  z,  w, -x],
        [ z, -y,  x,  w],
    ], dtype=float)

    omega_quat = np.array([0.0, wx, wy, wz], dtype=float)
    qdot = 0.5 * (H_r_plus @ omega_quat) + K_quat * quat_err * q
    return qdot  # (4,)

def quatdot_aux_casadi(q, omega, K_quat=0.0):
    q     = ca.reshape(ca.MX(q),     4, 1)
    omega = ca.reshape(ca.MX(omega), 3, 1)

    wx, wy, wz   = omega[0], omega[1], omega[2]
    ww           = 0.0

    quat_err = 1.0 - ca.dot(q, q)

    H_r_plus = ca.vertcat(
        ca.horzcat(ww, -wx, -wy, -wz),
        ca.horzcat(wx,  ww,  wz, -wy),
        ca.horzcat(wy, -wz,  ww,  wx),
        ca.horzcat(wz,  wy, -wx,  ww),
    )

    qdot = 0.5 * ca.mtimes(H_r_plus, q) + K_quat * quat_err * q
    return qdot

def quatdot_function():
    q_sym     = ca.MX.sym('q', 4)
    omega_sym = ca.MX.sym('omega', 3)
    K_sym     = ca.MX.sym('K', 1)
    ts = ca.MX.sym('ts', 1)

    qdot_sym  = quatdot_aux_casadi(q_sym, omega_sym, K_sym)
    dynamics_f = ca.Function('quatdot', [q_sym, omega_sym, K_sym], [qdot_sym])

    ## Integration method
    k1 = dynamics_f(q_sym, omega_sym, K_sym)
    k2 = dynamics_f(q_sym + (1/2)*ts*k1, omega_sym, K_sym)
    k3 = dynamics_f(q_sym + (1/2)*ts*k2, omega_sym, K_sym)
    k4 = dynamics_f(q_sym + ts*k3, omega_sym, K_sym)

    # Compute forward Euler method
    xk = q_sym + (1/6)*ts*(k1 + 2*k2 + 2*k3 + k4)
    #xk = x + ts*k1
    casadi_kutta = ca.Function('casadi_kutta',[q_sym, omega_sym, K_sym, ts], [xk])

    ## Calculate jacobian and gradient
    dfdx_f = ca.jacobian(xk, q_sym) 
    dfdu_f = ca.jacobian(xk, omega_sym)

    df_dx = ca.Function('df_dx', [q_sym, omega_sym, K_sym, ts], [dfdx_f])
    df_du = ca.Function('df_du', [q_sym, omega_sym, K_sym, ts], [dfdu_f])
    
    return casadi_kutta, df_dx, df_du

def integrate_gyro_quaternion(gyro, q0, t, dynamics, K_quat=10.0, renorm=True):
    gyro = np.asarray(gyro, float)
    t = np.asarray(t, float).reshape(-1)
    N = gyro.shape[1]
    assert gyro.shape == (3, N)
    assert t.shape[0] == N

    Q = np.zeros((4, N), dtype=float)
    q = np.asarray(q0, float).reshape(4)
    q /= np.linalg.norm(q)
    Q[:, 0] = q

    for k in range(N-1):
        dt = float(t[k+1] - t[k])
        omega = np.array([gyro[0, k], -gyro[1, k], -gyro[2, k]])
        aux = dynamics(q, omega, K_quat, dt)
        q = np.array(aux).reshape((4, ))
        if renorm:  # keep unit length
            q /= np.linalg.norm(q)

        Q[:, k+1] = q

    return Q

def simple_kalman_filter(gyro, q0, t, dynamics, A_d, B_d, rpy_acc, gain_q,
                         gain_r, K_quat=10.0):
    t = np.asarray(t, float).reshape(-1)
    H = np.eye(4, 4)

    Q = gain_q*np.eye(4, 4)
    R_m = gain_r*np.eye(4, 4)

    x = np.zeros((4, gyro.shape[1]))
    x[:, 0] = q0
    P = 1*np.eye(4, 4)

    for k in range(0, gyro.shape[1]-1):
        dt = float(t[k+1] - t[k])
        rot = R.from_euler('xyz', [rpy_acc[0, k], rpy_acc[1, k], rpy_acc[2, k]], degrees=False)
        q_xyzw = rot.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        z = q_wxyz.reshape((4,))
        
        A = A_d(x[:, k], gyro[:, k], 10, dt)
        xp = A@x[:, k]
        Pp = A@P@A.T + Q

        K = Pp@H.T@np.linalg.inv(H@Pp@H.T + R_m)
        
        aux = xp + K@(z - H@xp)
        x[:, k+1] = np.array(aux).reshape((4, ))
        P = Pp - K@H@Pp
    return x

def main():

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu_dir", default="../Data/Train/IMU/", help="Directory containing IMU files")
    parser.add_argument("--imu_file", default="imuRaw4", help="IMU file name (without extension)")
    parser.add_argument("--vicon_dir", default="../Data/Train/Vicon/", help="Directory containing Vicon files")
    parser.add_argument("--vicon_file", default="viconRot4", help="Vicon file name (without extension)")
    parser.add_argument("--imu_params", default="../IMUParams.mat", help="Path to IMU parameters file")
    args = parser.parse_args()

    # Full paths
    imu_path = Path(args.imu_dir) / f"{args.imu_file}.mat"
    vicon_path = Path(args.vicon_dir) / f"{args.vicon_file}.mat"
    params_path = Path(args.imu_params)

    # Load data
    imu = io.loadmat(imu_path)
    vicon = io.loadmat(vicon_path)
    params = io.loadmat(params_path)
    
    # Get IMU data
    imu_data = imu["vals"]
    imu_ts = imu["ts"]
    imu_ts = imu_ts
    
    # Get Vicon Data
    vicon_data = vicon["rots"]
    vicon_ts = vicon["ts"]
    vicon_ts = vicon_ts
    
    # Parameters of the system bias and scale
    imu_params = params["IMUParams"]

    # Scaled IMU Data
    acc_data_filtered, gyro_data_filtered = scale_measurements(imu_data, imu_params)

    # Check the common time grid 
    t_sync = _choose_target_time(imu_ts[0, :].ravel(), vicon_ts[0, :].ravel(), prefer="denser")

    # IMU (acc & gyro) are linear-interpolated per axis
    acc_sync  = interp_linear_timeseries(imu_ts[0, :],  acc_data_filtered,  t_sync)   
    gyro_sync = interp_linear_timeseries(imu_ts[0, :],  gyro_data_filtered, t_sync)   

    # Plot the signals accelerometers and gyroscope
    plot_acc(t_sync, acc_sync)
    plot_gyro(t_sync, gyro_sync)

    # Interpolation of the rotation matrices 
    R_sync = slerp_rotmats(vicon_ts[0, :], vicon_data, t_sync)

    # Compute euler angles from vicon
    quaternion_vicon_sync = quat_from_matrix(R_sync)
    rpy_vicon_sync = quat_to_euler_xyz(quaternion_vicon_sync)

    # Angles from acc
    rpy_acc_sync = angles_from_acc(acc_sync)

    # Dystem dynamics and A matrix
    dynamics, df_dx, df_du = quatdot_function()

    # Integral gyro using quaternion dot
    q0 = quaternion_vicon_sync[:, 0]
    q_gyro_sync = integrate_gyro_quaternion(gyro_sync, q0, t_sync, dynamics)
    rpy_gyro_quat = quat_to_euler_xyz(q_gyro_sync)

    q_complementary = simple_kalman_filter(gyro_sync, q0, t_sync, dynamics, df_dx, df_du,
                         rpy_acc_sync, 0.00001, 100000,  K_quat=10.0)
    rpy_complementary = quat_to_euler_xyz(q_complementary)

    ## Comparative results
    plot_all_methods(t_sync, rpy_acc_sync, t_sync, rpy_vicon_sync, t_sync,
                     rpy_gyro_quat, t_sync, rpy_complementary, "Results_4")
    
    # Video
    #make_orientation_video(t_sync, rpy_vicon_sync, rpy_acc_sync, rpy_gyro_quat,
                           #rpy_complementary, out_path="orientations.mp4", euler_order="xyz", degrees=False, fps=None)


if __name__ == "__main__":
    main()
