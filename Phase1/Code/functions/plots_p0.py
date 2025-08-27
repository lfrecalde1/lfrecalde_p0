import matplotlib.pyplot as plt
import scienceplots

def plot_samples(imu_ts):
    with plt.style.context(["science", "no-latex"]): 
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(imu_ts[0, :], label=f"ts")
        ax.set_xlabel("Samples [k]")
        ax.set_ylabel("Time [s]")
        ax.autoscale(tight=True)
        fig.savefig("sample_imu.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_acc(time_s, imu_data_filtered):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["x", "y", "z"]
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :], label=f"a{labels[i]}")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [m/s^2]")
        ax.autoscale(tight=True)
        fig.savefig("acc.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_gyro(time_s, imu_data_filtered):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["x", "y", "z"]
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :], label=f"w{labels[i]}")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular Velocity [rad/s]")
        ax.autoscale(tight=True)
        fig.savefig("gyro.pdf", dpi=300, bbox_inches="tight")
    return None

def plot_angles(time_s, imu_data_filtered, name):

    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["roll", "pitch", "yaw"]
        colors = ["red", "green", "blue"]  # x=roll → red, y=pitch → green, z=yaw → blue
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :],
                    label=f"{labels[i]}",
                    color=colors[i])
        ax.legend(loc="upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [rad]")
        ax.autoscale(tight=True)

        filename = f"angles_{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    return None

def plot_quaternions(time_s, quat_data, name):
    with plt.style.context(["science", "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 2))
        labels = ["w", "x", "y", "z"]
        colors = ["black", "red", "green", "blue"]  # distinct colors

        for i in range(quat_data.shape[0]):
            ax.plot(time_s, quat_data[i, :],
                    label=f"{labels[i]}",
                    color=colors[i])

        ax.legend(loc="upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Quaternion value")
        ax.autoscale(tight=True)

        filename = f"quaternions_{name}.pdf"
        fig.savefig(filename, dpi=300, bbox_inches="tight")

def plot_all_methods(time_acc, rpy_acc,
                     time_rot, rpy_rot,
                     time_gyro, rpy_gyro,
                     name="rpy_axis"):
    with plt.style.context(["science", "no-latex"]):
        labels = ["roll", "pitch", "yaw"]
        axis_colors = ["red", "green", "blue"]   # x=roll, y=pitch, z=yaw
        linestyles = {
            "acc":  "dashed",   # accelerometer
            "vicon":  "solid",    # vicon (rotation matrices)
            "gyro": "dotted"    # gyroscope
        }

        methods = [
            (time_acc,  rpy_acc,  "acc"),
            (time_rot,  rpy_rot,  "vicon"),
            (time_gyro, rpy_gyro, "gyro")
        ]

        for i, label in enumerate(labels):
            fig, ax = plt.subplots(figsize=(8, 3))

            for time_s, data, method in methods:
                # Vicon in black; others keep axis color
                color = "black" if method == "vicon" else axis_colors[i]
                ax.plot(
                    time_s, data[i, :],
                    label=f"{method}",
                    color=color,
                    linestyle=linestyles[method],
                    linewidth=1.5 if method == "vicon" else 1.0,
                    zorder=3 if method == "vicon" else 2
                )

            ax.set_title(f"{label.capitalize()} comparison")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Angle [rad]")
            ax.legend(loc="upper right")
            ax.autoscale(tight=True)

            filename = f"{name}_{label}.pdf"
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
