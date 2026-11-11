import time
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState

import tf2_ros


class FakeMoveItPickPlaceNode(Node):

    def __init__(self):
        super().__init__('fake_moveit_pick_place_node')

        self.busy = False
        self.done_once = False
        self.latest_joint_state = None
        self.saved_target = None
        self.initial_pose_done = False

        self.sim_gripper = True
        self.ik_link_name = 'link_eef'

        self.GRASP_X_OFFSET = 0.00
        self.GRASP_Y_OFFSET = 0.00
        self.GRASP_Z_OFFSET = 0.00

        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]

        self.initial_detect_pose = self.deg_list([
            0.7, 5.0, 4.7, 0.9, 8.0, -57.3, -1.2
        ])

        self.after_pick_pose = self.deg_list([
            -7.6, 49.9, 5.6, 32.3, 0.3, -83.7, -1.2
        ])

        self.coffee_machine_pose = self.deg_list([
            2.1, 56.0, 3.4, 55.5, -75.2, -73.3, -1.2
        ])

        self.consumer_place_pose = self.deg_list([
            -1.8, 1.4, 7.7, 2.0, 6.4, -87.6, -1.2
        ])

        self.GRIPPER_OPEN = 550.0
        self.GRIPPER_HOLD = 495.0

        self.callback_group = ReentrantCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )

        self.create_subscription(
            PointStamped,
            '/target_base',
            self.target_callback,
            10,
            callback_group=self.callback_group
        )

        self.marker_lock_pub = self.create_publisher(
            Bool,
            '/cup_marker_lock',
            10
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/xarm7_traj_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )

        self.create_timer(1.0, self.status_timer)

        self.get_logger().info("Waiting for /compute_ik service...")
        self.ik_client.wait_for_service()

        self.get_logger().info("Waiting for Gazebo arm trajectory action server...")
        self.arm_action_client.wait_for_server()

        self.get_logger().info("Fake MoveIt pick-place node started")
        self.get_logger().info("Gripper is FAKE/SIMULATED in this node")

    def lock_cup_marker(self):
        msg = Bool()
        msg.data = True
        self.marker_lock_pub.publish(msg)

        self.get_logger().warn(
            "Published /cup_marker_lock = True. Gazebo cup marker is locked."
        )

    def deg_list(self, values):
        return [math.radians(v) for v in values]

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def status_timer(self):
        if self.latest_joint_state is None:
            self.get_logger().warn("Still waiting for /joint_states...")
            return

        if not self.initial_pose_done and not self.busy:
            self.busy = True
            threading.Thread(
                target=self.move_to_initial_detection_pose,
                daemon=True
            ).start()
            return

        if (
            self.initial_pose_done
            and self.saved_target is not None
            and not self.busy
            and not self.done_once
        ):
            x, y, z = self.saved_target
            self.busy = True
            self.done_once = True

            self.lock_cup_marker()

            threading.Thread(
                target=self.pick_coffee_sequence,
                args=(x, y, z),
                daemon=True
            ).start()

    def target_callback(self, msg):
        if self.done_once:
            return

        x = msg.point.x
        y = msg.point.y
        z = msg.point.z

        if x < 0.10 or x > 0.80:
            self.get_logger().warn(f"Rejected X out of range: {x:.3f}")
            return

        if y < -0.40 or y > 0.40:
            self.get_logger().warn(f"Rejected Y out of range: {y:.3f}")
            return

        if z < 0.01 or z > 0.70:
            self.get_logger().warn(f"Rejected Z out of range: {z:.3f}")
            return

        self.saved_target = (x, y, z)

        self.get_logger().info(
            f"Target saved: X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

    def wait_future(self, future, timeout_sec):
        start_time = time.time()

        while rclpy.ok():
            if future.done():
                return True

            if time.time() - start_time > timeout_sec:
                return False

            time.sleep(0.05)

        return False

    def get_current_eef_orientation(self):
        for i in range(20):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'link_base',
                    self.ik_link_name,
                    rclpy.time.Time()
                )
                return transform.transform.rotation

            except Exception as e:
                self.get_logger().warn(
                    f"Waiting for link_base -> {self.ik_link_name} TF "
                    f"{i + 1}/20: {e}"
                )
                time.sleep(0.2)

        return None

    def compute_ik(self, x, y, z):
        if self.latest_joint_state is None:
            self.get_logger().warn("No /joint_states available for IK")
            return None

        orientation = self.get_current_eef_orientation()

        if orientation is None:
            self.get_logger().warn("Could not get end-effector orientation")
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'link_base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = orientation

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'xarm7'
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = False

        robot_state = RobotState()
        robot_state.joint_state = self.latest_joint_state
        req.ik_request.robot_state = robot_state

        self.get_logger().info(
            f"Calling IK for {self.ik_link_name}: "
            f"X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        future = self.ik_client.call_async(req)

        if not self.wait_future(future, 10.0):
            self.get_logger().warn("IK timeout")
            return None

        result = future.result()

        if result.error_code.val != 1:
            self.get_logger().warn(
                f"IK failed with error code: {result.error_code.val}"
            )
            return None

        joint_state = result.solution.joint_state
        joint_map = dict(zip(joint_state.name, joint_state.position))

        try:
            positions = [joint_map[name] for name in self.joint_names]

            current_joint_map = dict(
                zip(self.latest_joint_state.name, self.latest_joint_state.position)
            )

            current_positions = [
                current_joint_map[name] for name in self.joint_names
            ]

            max_allowed_change = 2.6

            for i in range(len(positions)):
                diff = abs(positions[i] - current_positions[i])

                if diff > max_allowed_change:
                    self.get_logger().warn(
                        f"IK rejected: joint{i + 1} changes too much: {diff:.2f} rad"
                    )
                    return None

            positions[6] = current_positions[6]

        except KeyError as e:
            self.get_logger().warn(f"IK solution missing joint: {e}")
            return None

        self.get_logger().info(f"IK solution accepted: {positions}")
        return positions

    def send_arm_goal(self, positions, duration=8):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration

        goal_msg.trajectory.points.append(point)
        goal_msg.goal_time_tolerance.sec = 3

        self.get_logger().info(f"Sending arm goal: {positions}")

        future = self.arm_action_client.send_goal_async(goal_msg)

        if not self.wait_future(future, 5.0):
            self.get_logger().warn("Arm goal send timeout")
            return False

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Arm goal rejected")
            return False

        self.get_logger().info("Arm goal accepted")

        result_future = goal_handle.get_result_async()

        if not self.wait_future(result_future, duration + 6):
            self.get_logger().warn("Arm result timeout, continuing")
            time.sleep(1.0)
            return True

        result = result_future.result().result

        if result.error_code == 0:
            self.get_logger().info("Arm motion complete")
            time.sleep(0.5)
            return True

        self.get_logger().warn(f"Arm goal failed: {result.error_string}")
        return False

    def send_gripper_goal(self, position):
        self.get_logger().info(
            f"[FAKE GRIPPER] Pretending gripper moved to {position}"
        )
        time.sleep(1.0)
        return True

    def open_gripper(self):
        self.get_logger().info("Opening fake gripper")
        return self.send_gripper_goal(self.GRIPPER_OPEN)

    def close_gripper(self):
        self.get_logger().info("Closing fake gripper")
        return self.send_gripper_goal(self.GRIPPER_HOLD)

    def move_to_pose(self, x, y, z, duration=8):
        joints = self.compute_ik(x, y, z)

        if joints is None:
            return False

        self.get_logger().info(
            f"Moving to pose X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        return self.send_arm_goal(joints, duration=duration)

    def move_to_initial_detection_pose(self):
        self.get_logger().info("Moving to initial detection pose...")

        if self.send_arm_goal(self.initial_detect_pose, duration=8):
            self.get_logger().info("Initial detection pose reached")
            self.initial_pose_done = True
        else:
            self.get_logger().warn("Failed to move to initial detection pose")

        self.busy = False

    def pick_coffee_sequence(self, x, y, z):
        self.get_logger().info(
            f"Starting FAKE simulated sequence at cup "
            f"X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        grip_x = x + self.GRASP_X_OFFSET
        grip_y = y + self.GRASP_Y_OFFSET
        grip_z = z + self.GRASP_Z_OFFSET

        above_z = grip_z + 0.10
        pick_z = grip_z + 0.02

        self.get_logger().info(
            f"Corrected grip target: "
            f"X={grip_x:.3f}, Y={grip_y:.3f}, Z={grip_z:.3f}"
        )

        if not self.open_gripper():
            self.busy = False
            return

        if not self.move_to_pose(grip_x, grip_y, above_z, duration=8):
            self.busy = False
            return

        if not self.move_to_pose(grip_x, grip_y, pick_z, duration=8):
            self.busy = False
            return

        if not self.close_gripper():
            self.busy = False
            return

        if not self.send_arm_goal(self.after_pick_pose, duration=10):
            self.busy = False
            return

        if not self.send_arm_goal(self.coffee_machine_pose, duration=10):
            self.busy = False
            return

        if not self.open_gripper():
            self.busy = False
            return

        self.get_logger().info("Fake cup sequence complete at coffee machine")
        time.sleep(3.0)

        if not self.send_arm_goal(self.consumer_place_pose, duration=10):
            self.busy = False
            return

        if not self.send_arm_goal(self.initial_detect_pose, duration=10):
            self.busy = False
            return

        self.get_logger().info("Full fake pick-place sequence complete")
        self.busy = False


def main(args=None):
    rclpy.init(args=args)

    node = FakeMoveItPickPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
