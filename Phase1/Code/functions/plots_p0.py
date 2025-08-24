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
        labels = ["roll", "pitch", "way"]
        for i in range(imu_data_filtered.shape[0]):
            ax.plot(time_s, imu_data_filtered[i, :], label=f"{labels[i]}")
        ax.legend(loc = "upper right")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [rad]")
        ax.autoscale(tight=True)
        name = "angles" + "_" + name + ".pdf"
        fig.savefig(name, dpi=300, bbox_inches="tight")
