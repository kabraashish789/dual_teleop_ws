# Dual Teleop Workspace

This is a ROS 2 Humble workspace for controlling dual Panda robot arms in rviz using teleoperation. It includes MoveIt 2 configuration, URDF descriptions, and teleoperation nodes that use webcam-based hand gesture control.

## Packages

- `dual_panda_description`: URDF and meshes for the robots
- `dual_panda_moveit_config`: MoveIt 2 configuration for dual arm planning
- `dual_panda_teleop`: Teleoperation nodes for controlling the Panda arms

## Prerequisites

### Install ROS 2 Humble

Follow the official ROS 2 Humble installation guide for Ubuntu:
[https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)

### Install MoveIt 2

Follow the official MoveIt 2 installation guide:
[https://moveit.picknik.ai/main/doc/tutorials/getting_started/getting_started.html](https://moveit.picknik.ai/main/doc/tutorials/getting_started/getting_started.html)

NOTE: Be sure to change any reference from ROS 2 `jazzy` to `humble` during the installation steps.

This setup will typically create a workspace like:
```
~/ws_moveit/
```

## Workspace Setup

### 1. Build the MoveIt workspace

```bash
cd ~/ws_moveit
colcon build
source install/setup.bash
```

### 2. Build the dual teleop workspace (overlay)

```bash
cd ~/dual_teleop_ws
colcon build
source install/setup.bash
```

> This creates an overlay workspace with `dual_teleop_ws` layered on top of `ws_moveit`.

## Running the System

### 1. Launch RViz with MoveIt demo

```bash
ros2 launch dual_panda_moveit_config demo.launch.py
```

### 2. Start webcam input

```bash
ros2 run usb_cam usb_cam_node_exe
```

### 3. Teleoperation Nodes

Two teleoperation modes are available:

#### a. Using Move Group

```bash
ros2 run dual_panda_teleop teleop_node
```

#### b. Using Joint Trajectory Publisher

```bash
ros2 run dual_panda_teleop teleop_trajectory
```

## Hand Gesture Control Mapping

- **Move hand up/down** → End-effector moves up/down
- **Move hand left/right** → End-effector moves left/right
- **Rotate hand sideways** → End-effector rotates
- **Left fist open/close** → Open/close left Panda gripper
- **Right fist open/close** → Open/close right Panda gripper

Left hand controls the **left arm**, right hand controls the **right arm**.

## Workspace Structure

```
dual_teleop_ws/
├── src/
│   ├── dual_panda_description/
│   ├── dual_panda_moveit_config/
│   └── dual_panda_teleop/
```

## Notes

- Always source `ws_moveit` first, then `dual_teleop_ws` to maintain overlay integrity
- Ensure your webcam is accessible to ROS (you may need permissions or to add your user to the `video` group)
- ROS 2 Humble and all dependencies (e.g. MoveIt, usb_cam) must be properly installed

## References

This project builds upon and is inspired by the following open-source repositories:

- [moveit/moveit_resources](https://github.com/moveit/moveit_resources)
- [e-candeloro/Vision-Robotic-Arm-Gesture-Recognition](https://github.com/e-candeloro/Vision-Robotic-Arm-Gesture-Recognition)

