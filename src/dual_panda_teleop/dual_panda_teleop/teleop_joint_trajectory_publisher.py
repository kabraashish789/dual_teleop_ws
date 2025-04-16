#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

class JointTrajectoryTeleopNode(Node):
    def __init__(self):
        super().__init__('joint_trajectory_teleop')

        # Publishers for each arm
        self.left_traj_pub = self.create_publisher(JointTrajectory, '/left_panda_arm_controller/joint_trajectory', 10)
        self.right_traj_pub = self.create_publisher(JointTrajectory, '/right_panda_arm_controller/joint_trajectory', 10)

        # Camera subscriber
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

        # Position buffers
        self.prev_positions = {'Left': None, 'Right': None}
        self.current_positions = [0.0, -2.356, 0.0, -2.356, 1.571, 1.571]  # [L-j1, L-j4, R-j1, R-j4, L-j6, R-j6]

    def clamp(self, val, min_val, max_val):
        return max(min(val, max_val), min_val)

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
                x = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                y = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y
                current_pos = np.array([x, y])

                prev = self.prev_positions[label]
                if prev is not None:
                    delta = current_pos - prev

                    # Adjust gains
                    H_GAIN = 2.0
                    V_GAIN = 2.0
                    R_GAIN = -2.0  # rotation gain for joint6

                    # Wrist and index tip
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    dx = index_tip.x - wrist.x
                    dy = index_tip.y - wrist.y
                    angle = np.arctan2(dy, dx)

                    if label == 'Left':
                        self.current_positions[0] += delta[0] * H_GAIN
                        self.current_positions[1] += -delta[1] * V_GAIN
                        self.current_positions.append(0.0) if len(self.current_positions) < 6 else None
                        self.current_positions[4] = angle * R_GAIN  # left_panda_joint6
                    elif label == 'Right':
                        self.current_positions[2] += delta[0] * H_GAIN
                        self.current_positions[3] += -delta[1] * V_GAIN
                        self.current_positions.append(0.0) if len(self.current_positions) < 6 else None
                        self.current_positions[5] = angle * R_GAIN  # right_panda_joint6

                self.prev_positions[label] = current_pos

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
        # Left arm trajectory
        left_traj = JointTrajectory()
        left_traj.joint_names = [
            'left_panda_joint1',
            'left_panda_joint2',
            'left_panda_joint3',
            'left_panda_joint4',
            'left_panda_joint5',
            'left_panda_joint6',
            'left_panda_joint7'
        ]
        left_point = JointTrajectoryPoint()
        left_point.positions = [
            self.current_positions[0],  # joint1 (controlled)
            -0.785,                     # joint2
            0.0,                        # joint3
            self.current_positions[1], # joint4 (controlled)
            0.0,                        # joint5
            self.current_positions[4],  # joint6
            0.785                      # joint7
        ]
        left_point.time_from_start.sec = 1
        left_traj.points.append(left_point)
        self.left_traj_pub.publish(left_traj)
        # Print left arm joint positions
        self.get_logger().info(f"[LEFT ARM] Sent trajectory:\n" +
            "\n".join([f"  {name}: {pos:.3f}" for name, pos in zip(left_traj.joint_names, left_point.positions)]))

        # Right arm trajectory
        right_traj = JointTrajectory()
        right_traj.joint_names = [
            'right_panda_joint1',
            'right_panda_joint2',
            'right_panda_joint3',
            'right_panda_joint4',
            'right_panda_joint5',
            'right_panda_joint6',
            'right_panda_joint7'
        ]
        right_point = JointTrajectoryPoint()
        right_point.positions = [
            self.current_positions[2],
            -0.785,
            0.0,
            self.current_positions[3],
            0.0,
            self.current_positions[5],
            0.785
        ]
        right_point.time_from_start.sec = 1
        right_traj.points.append(right_point)
        self.right_traj_pub.publish(right_traj)
        # Print right arm joint positions
        self.get_logger().info(f"[RIGHT ARM] Sent trajectory:\n" +
            "\n".join([f"  {name}: {pos:.3f}" for name, pos in zip(right_traj.joint_names, right_point.positions)]))

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


