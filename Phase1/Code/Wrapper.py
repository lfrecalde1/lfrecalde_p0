import argparse
from pathlib import Path
import numpy as np
from scipy import io
from functions.plots_p0 import plot_samples, plot_acc, plot_gyro, plot_angles, plot_all_methods
from scipy.spatial.transform import Rotation as R

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
        rpy[0, k] = np.arctan(acc[1, k]/(np.sqrt(acc[0, k]**2 + acc[2, k]**2)))
        rpy[1, k] = np.arctan(-acc[0, k]/(np.sqrt(acc[1, k]**2 + acc[2,
                                                                     k]**2)))
        rpy[2, k] = np.arctan((np.sqrt(acc[0, k]**2 + acc[1, k]**2))/acc[2, k])

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

    # Split Angles
    roll = euler[0, 0]
    pitch = euler[1, 0]
    yaw = euler[2, 0]

    # Create matrix for the mapping
    w11 = 1.0
    w12 = 0.0
    w13 = -np.sin(pitch)
    
    w21 = 0.0
    w22 = np.cos(roll)
    w23 = np.sin(roll)*np.cos(pitch)

    w31 = 0.0
    w32 = -np.sin(roll)
    w33 = np.cos(roll)*np.cos(pitch)

    W  = np.array([[w11, w12, w13], [w21, w22, w23], [w31, w32, w33]])
    
    # Reshape angular velocity body frame
    omega = omega.reshape((omega.shape[0], 1))
    
    # Ode equation
    f_dot =  np.linalg.inv(W)@omega
    return f_dot

def f_rk4(x, u, ts, dynamics):

    # Function which computes the runge kutta integration method.
    # INPUT
    # x                                          - states of the system x_k
    # u                                          - control actions of the system u_k
    # ts                                         - sample time
    # OUTPUT
    # x                                          - states of the system x_{k+1}
    # reshape States and control actions
    x = x.reshape((x.shape[0], 1))
    u = u.reshape((u.shape[0], 1))

    k1 = dynamics(x, u)
    k2 = dynamics(x + (1/2)*ts*k1, u)
    k3 = dynamics(x + (1/2)*ts*k2, u)
    k4 = dynamics(x + ts*k3, u)

    # Compute forward Euler method
    x = x + (1/6)*ts*(k1 + 2*k2 + 2*k3 + k4)
    x = x[:, 0]
    return x

def angles_from_gyro(gyro, rpy):
    
    # Initial Conditions
    rpy_estimated = np.zeros((3, gyro.shape[1]))
    rpy_estimated[:, 0] = rpy[:, 0]
    
    # Foor loop of the system
    for k in range(0, rpy_estimated.shape[1]-1):
        # Integration method
        rpy_estimated[:, k+1] = f_rk4(rpy_estimated[:, k], gyro[:, k], 0.01, euler_dot)

    return rpy_estimated

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
    imu_ts = imu_ts - imu_ts[0, 0]
    
    # Get Vicon Data
    vicon_data = vicon["rots"]
    vicon_ts = vicon["ts"]
    vicon_ts = vicon_ts - vicon_ts[0, 0]
    
    # Parameters of the system bias and scale
    imu_params = params["IMUParams"]

    # Scaled IMU Data
    acc_data_filtered, gyro_data_filtered = scale_measurements(imu_data, imu_params)
    
    # Sanity Check data
    plot_samples(imu_ts)
    plot_acc(imu_ts[0, :], acc_data_filtered)
    plot_gyro(imu_ts[0, :], gyro_data_filtered)
    
    # Compute Angles from Accelerometer
    rpy_acc = angles_from_acc(acc_data_filtered)
    plot_angles(imu_ts[0, :], rpy_acc, "acc_rpy")

    # Compute angles from rotation matrices
    rpy_rot = angles_from_rot(vicon_data)
    plot_angles(vicon_ts[0, :], rpy_rot, "rot_rpy")
    
    # Estimation based on the Gyroscope
    rpy_gyro = angles_from_gyro(gyro_data_filtered, rpy_rot)
    plot_angles(imu_ts[0, :], rpy_gyro, "gyro_rpy")

    # Compare methods 
    plot_all_methods(imu_ts[0, :], rpy_acc, vicon_ts[0, :], rpy_rot, imu_ts[0, :], rpy_gyro, "Comparative")
    
    # Check the outputs
    print(vicon_ts)
    print(imu_ts)

if __name__ == "__main__":
    main()
