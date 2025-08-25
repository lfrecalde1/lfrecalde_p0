import argparse
from pathlib import Path
import numpy as np
from scipy import io
from functions.plots_p0 import plot_samples, plot_acc, plot_gyro, plot_angles, plot_all_methods
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp

def _overlap_window(t1, t2):
    """Return [t0, t1] overlap window or raise if no overlap."""
    t0 = max(t1.min(), t2.min())
    t1_ = min(t1.max(), t2.max())
    if t1_ <= t0:
        raise ValueError("No time overlap between the two sequences.")
    return t0, t1_

def _median_dt(t):
    return float(np.median(np.diff(t)))

def _choose_target_time(t_imu, t_vicon, prefer="denser"):
    """
    Return target time vector restricted to overlap.
    prefer: 'denser' (auto), 'imu', or 'vicon'
    """
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
    """
    Linear interpolation per row.
    X_src: shape (D, N) or (N,) at t_src shape (N,)
    Returns X_tgt with shape (D, M) or (M,)
    """
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
    """
    SLERP interpolate rotation matrices along time.
    R_src_3x3xN: shape (3,3,N)
    Returns Rotation object evaluated at t_tgt and also returns Euler (xyz) if desired.
    """
    N = R_src_3x3xN.shape[2]
    # Build Rotation sequence
    Rs = R.from_matrix(np.transpose(R_src_3x3xN, (2, 0, 1)))  # (N,3,3)
    slerp = Slerp(np.asarray(t_src).ravel(), Rs)
    Rtgt = slerp(np.asarray(t_tgt).ravel())  # Rotation object at target times
    return Rtgt

def angles_from_rotation_obj_xyz(Robj):
    """Return rpy (xyz) angles from a Rotation object sequence (length M)."""
    # as_euler returns (M,3) with columns x,y,z
    eul = Robj.as_euler('xyz', degrees=False)
    return eul.T  # shape (3, M)

def scale_measurements(imu, parameters):
    """
    Map IMU raw values to real units.
    imu: shape (3, N)
    parameters: 2x3 matrix [[sx, sy, sz], [bx, by, bz]]
    Formula: (raw + bias) / scale
    """
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
    """
    Accelerometer to angles.
    acc: shape (3, N)
    """
    rpy = np.zeros_like(acc)

    for k in range(0, rpy.shape[1]):
        rpy[2, k] = np.arctan((np.sqrt(acc[0, k]**2 + acc[1, k]**2))/acc[2, k])
        rpy[0, k] = np.arctan2(acc[1, k], acc[2, k])
        rpy[1, k] = -np.arctan2(acc[0, k], np.sqrt(acc[1, k]**2 + acc[2, k]**2))

    return rpy

def angles_from_rot(rot):
    """
    Rotational Matrix to Euler angles.
    """
    rpy = np.zeros((3, rot.shape[2]))
    for k in range(0, rot.shape[2]):
        r = R.from_matrix(rot[:, :, k])
        rpy[:, k] = r.as_euler('xyz', degrees=False)
    return rpy

def euler_dot(euler, omega):
    roll, pitch, yaw = euler.flatten()
    W = np.array([
        [1.0, 0.0, -np.sin(pitch)],
        [0.0, np.cos(roll),  np.sin(roll)*np.cos(pitch)],
        [0.0, -np.sin(roll), np.cos(roll)*np.cos(pitch)]
    ])
    omega = omega.reshape(3, 1)
    return np.linalg.inv(W) @ omega  # (3,1)

def integrate_gyro_euler(gyro, rpy0, t):
    """
    Integrate Euler angles from body rates using RK4 with variable time steps.
    gyro: (3, M) at times t (M,)
    rpy0: (3,) initial angle at t[0]
    t:   (M,)
    """
    M = gyro.shape[1]
    rpy = np.zeros((3, M))
    rpy[:, 0] = rpy0
    for k in range(M-1):
        dt = (t[k+1] - t[k])
        x  = rpy[:, k].reshape(3, 1)
        u  = gyro[:, k].reshape(3, 1)

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

def main():

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu_dir", default="../Data/Train/IMU/", help="Directory containing IMU files")
    parser.add_argument("--imu_file", default="imuRaw1", help="IMU file name (without extension)")
    parser.add_argument("--vicon_dir", default="../Data/Train/Vicon/", help="Directory containing Vicon files")
    parser.add_argument("--vicon_file", default="viconRot1", help="Vicon file name (without extension)")
    parser.add_argument("--imu_params", default="../IMUParams.mat", help="Path to IMU parameters file")
    args = parser.parse_args()

    # Full paths
    imu_path = Path(args.imu_dir) / f"{args.imu_file}.mat"
    vicon_path = Path(args.vicon_dir) / f"{args.vicon_file}.mat"
    params_path = Path(args.imu_params)
    print(imu_path)
    print(vicon_path)

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

    # Vicon: SLERP rotations to t_sync, then convert to Euler xyz
    R_sync = slerp_rotmats(vicon_ts[0, :], vicon_data, t_sync)           
    rpy_vicon_sync = angles_from_rotation_obj_xyz(R_sync)
    
    # Angles from acc
    rpy_acc_sync = angles_from_acc(acc_sync)                              
    # --- integrate gyro on the same grid (variable dt) ---
    rpy0 = rpy_vicon_sync[:, 0]
    rpy_gyro_sync = integrate_gyro_euler(gyro_sync, rpy0, t_sync)        

    # Complementary Filter
    acc_sync_filter = low_passs_filter(acc_sync, 0.8)
    gyro_sync_filter = high_pass_filter(gyro_sync, 0.05, 0.01)
    
    # Integrate Gyro filter
    rpy_gyro_sync_filter = integrate_gyro_euler(gyro_sync_filter, rpy0, t_sync)

    # --- now plot on the same time base t_sync ---
    plot_angles(t_sync, rpy_acc_sync,  "acc_rpy_sync")
    plot_angles(t_sync, rpy_vicon_sync, "rot_rpy_sync")
    plot_angles(t_sync, rpy_gyro_sync, "gyro_rpy_sync")
    
    # Filter Signal
    plot_angles(t_sync, acc_sync_filter,  "acc_sync_filter")
    plot_angles(t_sync, acc_sync,  "acc_sync")

    plot_angles(t_sync, gyro_sync_filter,  "gyro_sync_filter")
    plot_angles(t_sync, gyro_sync,  "gyro_sync")

    # or comparative (all methods in one fig) or per-axis figs if you prefer:
    plot_all_methods(t_sync, rpy_acc_sync, t_sync, rpy_vicon_sync, t_sync, rpy_gyro_sync, "Comparative_sync")
    plot_all_methods(t_sync, rpy_acc_sync, t_sync, rpy_vicon_sync, t_sync,
                     rpy_gyro_sync_filter, "Comparative_sync_filter")

if __name__ == "__main__":
    main()
