# 🚁 Aerial Robotics - P0

This repository provides a simulation framework for a **quadrotor drone** navigating along a 3D infinity trajectory.  
The assignment focuses on **mission-level decision making** (when to land, when to take off, and how to resume waypoint navigation).  
Low-level control and physics are already implemented for you.

---

## 📂 Directory Structure

```text
P0/
│
├── main.py
├── generate_waypoints.py
├── waypoints.csv
│
├── utils.py
├── splat_render.py
│
├── quad_dynamics.py
├── drone.py
│
├── control.c
├── control.cpython-311-x86_64-linux-gnu.so
├── setup.py
├── build/
│
├── wb_frames_colmap/
├── wb_frames_splat/
```

## 📝 File-by-File Explanation

### 🔧 
- **`main.py`** → Main simulation script.  
  - Loads waypoints.  
  - Runs the guidance loop (mission logic).  
  - Calls the controller and physics engine.  
  - **👉 Students edit this file (see TODOs).**

- **`generate_waypoints.py`** → Generates an infinity-shaped trajectory and writes `waypoints.csv`.  
  - Includes plotting functions to visualize the path.  
  - Run this first to create your waypoint file.

- **`waypoints.csv`** → Waypoint list in **NED** (North–East–Down) coordinates.  
  - Format:
    ```csv
    N,E,D
    0.10,0.05,0.20
    0.20,0.15,0.18
    ...
    ```

### ⚙️ Support Code
- **`utils.py`** → Helper utilities for configuration, plotting, logging, and saving videos.  
- **`splat_render.py`** → Rendering backend using GSplat/Nerfstudio to produce visual outputs of the drone’s perspective.  

### 🪂 Drone Physics & Control
- **`quad_dynamics.py`** → Models the quadrotor dynamics (forces, torques, motion equations). **DO NOT EDIT.**  
- **`drone.py`** → Contains drone parameters (mass, inertia, rotor setup). **DO NOT EDIT.**

### 🛠️ Compiled Controller
- **`control.c` / `control.cpython-311-...so` / `setup.py` / `build/`**  
  - Precompiled low-level feedback controller.  
  - Converts guidance setpoints into motor thrusts.  
  - **DO NOT EDIT these files.**

### 🌍 Scene Data
- **`wb_frames_colmap/` & `wb_frames_splat/`** → Environment and rendering assets. Provided as-is.

---

## 🧭 Coordinate System

We use the **NED frame (North–East–Down):**

- **N** → North (x-axis)  
- **E** → East (y-axis)  
- **D** → Down (z-axis, positive = downward)  

Example: `(N=1.0, E=0.5, D=0.2)` = drone is 1 m north, 0.5 m east, 0.2 m below takeoff height.

---

## 🔄 System Architecture

The simulation runs in **three layers**:

1. **Guidance (high-level)**  
   - Decides what the drone *should do*:  
     - Navigate waypoints (Mode.NAV)  
     - Land (Mode.LANDING)  
     - Take off again (Mode.TAKEOFF)  
     - Finish mission (Mode.DONE)  
   - 👉 **Students implement this logic in `main.py`.**

2. **Control (low-level)**  
   - Converts guidance setpoints into motor thrusts.  
   - Already implemented in compiled C controller (`control.c`).  
   - **Do not edit.**

3. **Dynamics (physics)**  
   - Simulates the motion of the drone given thrust and torques.  
   - Implemented in `quad_dynamics.py`.  
   - **Do not edit.**

---

## 🎯 Student Tasks

You will **modify `main.py` and `utils.py`'**.

In `main.py` do the following,

1. **Waypoint Reading**
   - Implement CSV loading for `waypoints.csv`.  
   - Ensure you handle NED coordinates correctly.  

2. **Landing/Takeoff Logic**
   - Implement conditions for switching between modes:  
     - NAV → LANDING  
     - LANDING → TAKEOFF  
     - TAKEOFF → NAV  
   - The drone should land **once only** during the mission.  

3. **Continue Navigation**
   - After takeoff, the drone should resume waypoint following from where it left off.  

In `utils.py`, do the following

1. **Video Generation**
    - Read depth frames, rgb frames and stored waypoints
    - Create a video showing the RGB frame, depth frame and the 3D trajectory followed by the drone and the Groundtruth waypoints. The ground-truth waypoints should be in green while the drone trajectory should be in blue. When drone starts landing, the landing part of the trajectory should be in Red.

---

## 🚀 How to Run

python main.py
