#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import math


class JointTrajectoryTeleopNode(Node):
    def __init__(self):
        super().__init__('joint_trajectory_teleop')

        # Publishers for each arm
        self.left_traj_pub = self.create_publisher(JointTrajectory, '/left_panda_arm_controller/joint_trajectory', 10)
        self.right_traj_pub = self.create_publisher(JointTrajectory, '/right_panda_arm_controller/joint_trajectory', 10)

        # Gripper action clients
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_panda_arm_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_panda_arm_gripper_controller/gripper_cmd')

        self.last_gripper_state = {'Left': None, 'Right': None}

        # Camera subscriber
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

        # Joint positions: [L-j1, L-j4, R-j1, R-j4, L-j6, R-j6]
        self.current_positions = [0.0, -2.356, 0.0, -2.356, 1.571, 1.571]

        # Smoothing buffers
        self.position_buffer = {'Left': [], 'Right': []}
        self.window_size = 5  # moving average window

    def clamp(self, val, min_val, max_val):
        return max(min(val, max_val), min_val)

    def is_fist(self, landmarks):
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        tip_ids = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        distances = [math.hypot(landmarks.landmark[tip].x - wrist.x,
                                landmarks.landmark[tip].y - wrist.y) for tip in tip_ids]
        avg_dist = sum(distances) / len(distances)
        return avg_dist < 0.1

    def send_gripper_command(self, client, position):
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Gripper action server not available.")
            return
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 5.0
        client.send_goal_async(goal)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        image_rgb = cv2.cvtColor(cv2.flip(cv_image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                wrist_x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                wrist_y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                current_pos = np.array([wrist_x, wrist_y])

                # Update buffer
                self.position_buffer[label].append(current_pos)
                if len(self.position_buffer[label]) > self.window_size:
                    self.position_buffer[label].pop(0)

                # Compute smoothed delta
                if len(self.position_buffer[label]) >= 2:
                    prev_avg = np.mean(self.position_buffer[label][:-1], axis=0)
                    curr_avg = np.mean(self.position_buffer[label], axis=0)
                    delta = curr_avg - prev_avg

                    # Gains
                    H_GAIN = 2.0
                    V_GAIN = 2.0
                    R_GAIN = -2.0  # wrist rotation sensitivity

                    # Wrist angle for joint6 control
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    dx = index_tip.x - wrist.x
                    dy = index_tip.y - wrist.y
                    angle = np.arctan2(dy, dx)
                    rotation_value = angle * R_GAIN

                    if label == 'Left':
                        self.current_positions[0] += delta[0] * H_GAIN
                        self.current_positions[1] += -delta[1] * V_GAIN
                        self.current_positions[4] = rotation_value
                    elif label == 'Right':
                        self.current_positions[2] += delta[0] * H_GAIN
                        self.current_positions[3] += -delta[1] * V_GAIN
                        self.current_positions[5] = rotation_value

                # Gripper logic
                gripper_client = self.left_gripper_client if label == 'Left' else self.right_gripper_client
                is_hand_closed = self.is_fist(hand_landmarks)
                state_str = 'Closed' if is_hand_closed else 'Open'
                position = 0.0 if is_hand_closed else 0.0350

                if self.last_gripper_state[label] != state_str:
                    self.send_gripper_command(gripper_client, position)
                    self.last_gripper_state[label] = state_str
                    self.get_logger().info(f"{label} hand → Gripper {state_str}")

        # Clamp joint limits
        self.current_positions[0] = self.clamp(self.current_positions[0], -2.896, 2.896)  # L-j1
        self.current_positions[1] = self.clamp(self.current_positions[1], -3.071, 0.017)  # L-j4
        self.current_positions[2] = self.clamp(self.current_positions[2], -2.896, 2.896)  # R-j1
        self.current_positions[3] = self.clamp(self.current_positions[3], -3.071, 0.017)  # R-j4
        self.current_positions[4] = self.clamp(self.current_positions[4], -0.017, 3.752)  # L-j6
        self.current_positions[5] = self.clamp(self.current_positions[5], -0.017, 3.752)  # R-j6

        self.publish_trajectories()

        cv2.imshow("Teleoperation View", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Exiting Teleop")
            rclpy.shutdown()
            cv2.destroyAllWindows()

    def publish_trajectories(self):
        left_traj = JointTrajectory()
        left_traj.joint_names = [
            'left_panda_joint1', 'left_panda_joint2', 'left_panda_joint3',
            'left_panda_joint4', 'left_panda_joint5', 'left_panda_joint6', 'left_panda_joint7'
        ]
        left_point = JointTrajectoryPoint()
        left_point.positions = [
            self.current_positions[0], -0.785, 0.0,
            self.current_positions[1], 0.0, self.current_positions[4], 0.785
        ]
        left_point.time_from_start.sec = 1
        left_traj.points.append(left_point)
        self.left_traj_pub.publish(left_traj)
        self.get_logger().info(f"[LEFT ARM] Sent trajectory:\n" +
                               "\n".join([f"  {n}: {p:.3f}" for n, p in zip(left_traj.joint_names, left_point.positions)]))

        right_traj = JointTrajectory()
        right_traj.joint_names = [
            'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3',
            'right_panda_joint4', 'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7'
        ]
        right_point = JointTrajectoryPoint()
        right_point.positions = [
            self.current_positions[2], -0.785, 0.0,
            self.current_positions[3], 0.0, self.current_positions[5], 0.785
        ]
        right_point.time_from_start.sec = 1
        right_traj.points.append(right_point)
        self.right_traj_pub.publish(right_traj)
        self.get_logger().info(f"[RIGHT ARM] Sent trajectory:\n" +
                               "\n".join([f"  {n}: {p:.3f}" for n, p in zip(right_traj.joint_names, right_point.positions)]))


def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

