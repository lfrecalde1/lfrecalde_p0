import argparse
from pathlib import Path
import numpy as np
from scipy import io
from functions.plots_p0 import plot_samples, plot_acc, plot_gyro, plot_angles

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

    # Load data
    imu = io.loadmat(imu_path)
    vicon = io.loadmat(vicon_path)
    params = io.loadmat(params_path)
    
    # Get IMU data
    imu_data = imu["vals"]
    imu_ts = imu["ts"]* 1e-9
    imu_ts = imu_ts - imu_ts[0, 0]
    
    # Get Vicon Data
    vicon_data = vicon["rots"]
    vicon_ts = vicon["ts"]* 1e-9
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
    rpy = angles_from_acc(acc_data_filtered)
    plot_angles(imu_ts[0, :], rpy, "acc")

if __name__ == "__main__":
    main()
