#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from builtin_interfaces.msg import Duration
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose as GeoPose
import time


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleoperation_node')

        self.left_pose_pub = self.create_publisher(PoseStamped, '/left_panda_arm/target_pose', 10)
        self.right_pose_pub = self.create_publisher(PoseStamped, '/right_panda_arm/target_pose', 10)

        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

        self.last_publish_time = time.time()
        self.publish_interval = 2.0  # seconds between target updates

    def image_callback(self, msg: Image):
        current_time = time.time()
        if current_time - self.last_publish_time < self.publish_interval:
            return

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
                controlling_arm = "left_panda_arm" if label == "Right" else "right_panda_arm"
                ee_frame = "left_panda_hand" if controlling_arm == "left_panda_arm" else "right_panda_hand"

                avg_y = np.mean([lm.y for lm in hand_landmarks.landmark])
                avg_z = np.mean([lm.z for lm in hand_landmarks.landmark])  # Negative is toward camera

                base_y_offset = -1.5 if controlling_arm == "left_panda_arm" else 1.5

                target_pose = PoseStamped()
                target_pose.header.frame_id = 'world'

                # Use Y/Z from hand; get current X from TF
                try:
                    tf: TransformStamped = self.tf_buffer.lookup_transform(
                        'world',
                        ee_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    target_pose.pose.position.x = tf.transform.translation.x
                except TransformException as ex:
                    self.get_logger().warn(f'Could not get current X from TF: {ex}')
                    target_pose.pose.position.x = 0.5  # fallback

                target_pose.pose.position.y = base_y_offset + (0.5 - avg_y) * 0.4
                target_pose.pose.position.z = 0.3 + (-avg_z) * 0.2

                # Fixed orientation
                target_pose.pose.orientation.x = 0.0
                target_pose.pose.orientation.y = 0.0
                target_pose.pose.orientation.z = 0.0
                target_pose.pose.orientation.w = 1.0

                # Debug output
                print(f"[{controlling_arm.upper()}] Target Pose → X: {target_pose.pose.position.x:.3f}, "
                      f"Y: {target_pose.pose.position.y:.3f}, Z: {target_pose.pose.position.z:.3f}")

                if controlling_arm == "left_panda_arm":
                    self.left_pose_pub.publish(target_pose)
                else:
                    self.right_pose_pub.publish(target_pose)

                self.send_move_group_goal(controlling_arm, ee_frame, target_pose)
                self.last_publish_time = time.time()

        cv2.imshow("Teleoperation View", cv_image)
        cv2.waitKey(1)

    def send_move_group_goal(self, group_name, ee_frame, target_pose: PoseStamped):
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available!")
            return

        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                'world',
                ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except TransformException as ex:
            self.get_logger().warn(f'Could not get current EE pose: {ex}')
            return

        current_pose = GeoPose()
        current_pose.position.x = tf.transform.translation.x
        current_pose.position.y = tf.transform.translation.y
        current_pose.position.z = tf.transform.translation.z
        current_pose.orientation = target_pose.pose.orientation

        # Interpolate steps
        steps = 10
        poses = []
        for i in range(1, steps + 1):
            ratio = i / steps
            interp = GeoPose()
            interp.position.x = current_pose.position.x
            interp.position.y = current_pose.position.y + ratio * (target_pose.pose.position.y - current_pose.position.y)
            interp.position.z = current_pose.position.z + ratio * (target_pose.pose.position.z - current_pose.position.z)
            interp.orientation = target_pose.pose.orientation
            poses.append(interp)

        for pose in poses:
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = group_name
            goal_msg.request.num_planning_attempts = 3
            goal_msg.request.allowed_planning_time = 3.0
            goal_msg.request.start_state.is_diff = True

            pos_constraint = PositionConstraint()
            pos_constraint.header.frame_id = target_pose.header.frame_id
            pos_constraint.link_name = ee_frame

            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [0.01, 0.01, 0.01]

            pos_constraint.constraint_region.primitives.append(box)
            pos_constraint.constraint_region.primitive_poses.append(pose)
            pos_constraint.weight = 1.0

            orientation_constraint = OrientationConstraint()
            orientation_constraint.header.frame_id = target_pose.header.frame_id
            orientation_constraint.link_name = ee_frame
            orientation_constraint.orientation = pose.orientation
            orientation_constraint.absolute_x_axis_tolerance = 0.2
            orientation_constraint.absolute_y_axis_tolerance = 0.2
            orientation_constraint.absolute_z_axis_tolerance = 0.2
            orientation_constraint.weight = 1.0

            constraints = Constraints()
            constraints.position_constraints.append(pos_constraint)
            constraints.orientation_constraints.append(orientation_constraint)
            goal_msg.request.goal_constraints.append(constraints)

            self.move_group_client.send_goal_async(goal_msg)


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

