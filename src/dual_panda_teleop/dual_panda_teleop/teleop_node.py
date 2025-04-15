#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from control_msgs.action import GripperCommand
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import math
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose as GeoPose

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleoperation_node')

        # Action Clients
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_panda_arm_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_panda_arm_gripper_controller/gripper_cmd')

        # Camera subscriber
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        # MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)

        # Buffers
        self.hand_pos_buffer = {'Left': [], 'Right': []}
        self.fist_buffer = {'Left': [], 'Right': []}
        self.smooth_window = 5
        self.last_gripper_state = {'Left': None, 'Right': None}

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                group_name = 'left_panda_arm' if label == 'Right' else 'right_panda_arm'
                ee_frame = 'left_panda_hand' if label == 'Right' else 'right_panda_hand'
                gripper_client = self.left_gripper_client if label == 'Right' else self.right_gripper_client

                avg_y = np.mean([lm.y for lm in hand_landmarks.landmark])
                avg_z = np.mean([lm.z for lm in hand_landmarks.landmark])

                self.hand_pos_buffer[label].append((avg_y, avg_z))
                if len(self.hand_pos_buffer[label]) > self.smooth_window:
                    self.hand_pos_buffer[label].pop(0)

                smooth_y = np.mean([p[0] for p in self.hand_pos_buffer[label]])
                smooth_z = np.mean([p[1] for p in self.hand_pos_buffer[label]])

                delta_y = (0.5 - smooth_y) * 0.2
                delta_z = (-smooth_z) * 0.5

                base_position = {'left_panda_arm': [-0.5, -1.5, 0.7], 'right_panda_arm': [-0.5, 1.5, 0.7]}
                base_x, base_y, base_z = base_position[group_name]

                target_pose = PoseStamped()
                target_pose.header.frame_id = 'world'
                target_pose.pose.position.x = base_x
                target_pose.pose.position.y = base_y + delta_y
                target_pose.pose.position.z = max(0.1, base_z + delta_z)

                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                dx = index_tip.x - wrist.x
                dy = index_tip.y - wrist.y
                angle_z = math.atan2(dy, dx)

                target_pose.pose.orientation.x = 0.0
                target_pose.pose.orientation.z = 0.0
                target_pose.pose.orientation.y = math.sin(angle_z / 2.0)
                target_pose.pose.orientation.w = math.cos(angle_z / 2.0)

                # DEBUG print
                print(f"[{group_name.upper()}] Target Pose → X: {target_pose.pose.position.x:.3f}, "
                      f"Y: {target_pose.pose.position.y:.3f}, Z: {target_pose.pose.position.z:.3f}, "
                      f"AngleZ(rad): {angle_z:.2f}")

                # Send motion planning goal
                self.send_move_group_goal(group_name, ee_frame, target_pose)

                # Gripper control
                is_fist = self.is_fist(hand_landmarks)
                self.fist_buffer[label].append(is_fist)
                if len(self.fist_buffer[label]) > self.smooth_window:
                    self.fist_buffer[label].pop(0)
                smoothed_fist = np.mean(self.fist_buffer[label]) > 0.5
                self.handle_gripper(label, smoothed_fist, gripper_client)

        cv2.imshow("Teleoperation View", cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def send_move_group_goal(self, group_name, ee_frame, target_pose):
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(f"MoveGroup action server not available for {group_name}")
            return

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = group_name
        goal_msg.request.num_planning_attempts = 3
        goal_msg.request.allowed_planning_time = 2.0
        goal_msg.request.start_state.is_diff = True

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target_pose.header.frame_id
        pos_constraint.link_name = ee_frame
        box = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.02, 0.02, 0.02])
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = target_pose.header.frame_id
        orientation_constraint.link_name = ee_frame
        orientation_constraint.orientation = target_pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal_msg.request.goal_constraints.append(constraints)

        future = self.move_group_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('MoveGroup goal rejected.')
            return

        self.get_logger().info('MoveGroup goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Planning result code: {result.error_code.val}")

    def handle_gripper(self, hand_label, is_fist, client):
        state = 'Closed' if is_fist else 'Open'
        position = 0.0 if is_fist else 0.0350
        last_state = self.last_gripper_state[hand_label]

        if last_state != state:
            self.send_gripper_command(client, position)
            self.last_gripper_state[hand_label] = state
            self.get_logger().info(f"{hand_label} hand → Gripper {state}")

    def send_gripper_command(self, client, position):
        if not client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Gripper action server not available.")
            return
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 10.0
        client.send_goal_async(goal_msg)

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

def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

