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
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from xarm_msgs.srv import GripperMove, SetInt16, Call as XArmCall
from projv1_interfaces.srv import VisionDetect

import tf2_ros


class MoveItPickPlaceNode(Node):

    def __init__(self):
        super().__init__('moveit_pick_place_node')

        self.busy = False
        self.done_once = False
        self.latest_joint_state = None
        self.saved_target = None
        self.initial_pose_done = False
        self.gripper_ready = False
        self.allow_target_lock = False
        self.target_locked = False

        # VisionV1 one-shot service settings
        self.vision_requested = False
        self.VISION_SELECTED_CLASS = "Browncup"
        self.VISION_TARGET_FRAME = "link_base"
        self.VISION_WARMUP_FRAMES = 10

        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'
        ]

        self.initial_pose = self.deg_list([
            0, 0, 0, 0, 3.0, -90, 0
        ])


        # Initial detection pose from your xArm GUI
        self.initial_detect_pose = self.deg_list([
            0, 0, 0, 0, 3.0, -46, 0
        ])

        self.after_pick_pose = self.deg_list([
            -2.2, 54.8, 0, 48.3, 0.5, -94.1, 0
        ])

        self.coffee_machine_pose = self.deg_list([
            -0.9, 56.0, 3.4, 55.5, -75.2, -75.3, 0
        ])

        self.consumer_place_pose = self.deg_list([
            -6.8, 58.3, 2.6, 55.2, 40, -80.3, 0
        ])
        
        self.after_place_pose = self.deg_list([
            -6.8, 58.3, 2.6, 55.2, 40, -90.3, 0
        ])

        # xArm gripper raw values
        # 550 = open, 490 = close
        self.GRIPPER_OPEN = 800.0
        self.GRIPPER_HOLD = 480.0
        # Y-axis correction for left/right cup position
        # Increase Y_SIDE_SCALE if the arm does not move far enough left/right.
        # Decrease it if the arm moves too far left/right.
        self.Y_SIDE_SCALE = 1.10
        # Constant Y offset. Tune this if the gripper is always slightly left/right.
        self.Y_PICK_OFFSET = 0.0
        # X correction when cup is away from centre
        # Used because X is accurate at centre but slightly off at left/right.
        self.X_PICK_OFFSET = 0.035
        # Increase this if X error becomes larger when cup is more left/right.
        # Start small.
        self.X_SIDE_GAIN = 0.05


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

        # Gripper services that worked manually
        self.gripper_position_client = self.create_client(
            GripperMove,
            '/xarm/set_gripper_position',
            callback_group=self.callback_group
        )

        self.gripper_clean_client = self.create_client(
            XArmCall,
            '/xarm/clean_gripper_error',
            callback_group=self.callback_group
        )

        self.gripper_enable_client = self.create_client(
            SetInt16,
            '/xarm/set_gripper_enable',
            callback_group=self.callback_group
        )

        self.gripper_mode_client = self.create_client(
            SetInt16,
            '/xarm/set_gripper_mode',
            callback_group=self.callback_group
        )

        self.vision_client = self.create_client(
            VisionDetect,
            '/visionv1/detect',
            callback_group=self.callback_group
        )

        self.create_timer(1.0, self.status_timer)

        self.get_logger().info("Waiting for /compute_ik service...")
        self.ik_client.wait_for_service()

        self.get_logger().info("Waiting for arm trajectory action server...")
        self.arm_action_client.wait_for_server()

        self.get_logger().info("Waiting for gripper services...")
        self.gripper_position_client.wait_for_service()
        self.gripper_clean_client.wait_for_service()
        self.gripper_enable_client.wait_for_service()
        self.gripper_mode_client.wait_for_service()

        self.get_logger().info("Waiting for /visionv1/detect service...")
        self.vision_client.wait_for_service()

        self.get_logger().info("MoveIt coffee sequence node started")

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
            and not self.busy
            and not self.done_once
            and not self.vision_requested
        ):
            self.busy = True
            self.vision_requested = True

            threading.Thread(
                target=self.request_vision_and_pick,
                daemon=True
            ).start()

    def target_callback(self, msg):
        if self.done_once:
            return
            
        if self.target_locked:
            return
            
        if not self.allow_target_lock:
            return

        x = msg.point.x
        y = msg.point.y
        z = msg.point.z

        if x < 0.10 or x > 0.75:
            self.get_logger().warn(f"Rejected X out of range: {x:.3f}")
            return

        if y < -0.25 or y > 0.25:
            self.get_logger().warn(f"Rejected Y out of range: {y:.3f}")
            return

        if z < 0.03 or z > 0.40:
            self.get_logger().warn(f"Rejected Z out of range: {z:.3f}")
            return

        self.saved_target = (x, y, z)
        self.target_locked = True

        self.get_logger().info(
            f"Target LOCKED: X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )
        
        self.get_logger().info(
            "Future target updates will be ignored for accuracy."
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

    def call_empty_service(self, client, name, timeout=5.0):
        req = XArmCall.Request()
        future = client.call_async(req)

        if not self.wait_future(future, timeout):
            self.get_logger().warn(f"{name} service timeout")
            return False

        result = future.result()

        if result.ret != 0:
            self.get_logger().warn(
                f"{name} failed: ret={result.ret}, message={result.message}"
            )
            return False

        self.get_logger().info(f"{name} OK: {result.message}")
        return True

    def call_set_int16_service(self, client, value, name, timeout=5.0):
        req = SetInt16.Request()
        req.data = int(value)

        future = client.call_async(req)

        if not self.wait_future(future, timeout):
            self.get_logger().warn(f"{name} service timeout")
            return False

        result = future.result()

        if result.ret != 0:
            self.get_logger().warn(
                f"{name} failed: ret={result.ret}, message={result.message}"
            )
            return False

        self.get_logger().info(f"{name} OK: {result.message}")
        return True

    def setup_gripper(self):
        self.get_logger().info("Setting up gripper...")

        if not self.call_empty_service(
            self.gripper_clean_client,
            "clean_gripper_error"
        ):
            return False

        if not self.call_set_int16_service(
            self.gripper_enable_client,
            1,
            "set_gripper_enable"
        ):
            return False

        if not self.call_set_int16_service(
            self.gripper_mode_client,
            0,
            "set_gripper_mode"
        ):
            return False

        self.gripper_ready = True
        self.get_logger().info("Gripper setup complete")
        return True

    def get_current_eef_orientation(self):
        for i in range(20):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'link_base',
                    'link_tcp',
                    rclpy.time.Time()
                )
                return transform.transform.rotation

            except Exception as e:
                self.get_logger().warn(
                    f"Waiting for link_base -> link_tcp TF... {i + 1}/20: {e}"
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
        req.ik_request.ik_link_name = 'link_tcp'
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = False

        robot_state = RobotState()
        robot_state.joint_state = self.latest_joint_state
        req.ik_request.robot_state = robot_state

        self.get_logger().info(
            f"Calling IK for X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
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

            max_allowed_change = 1.5

            for i in range(len(positions)):
                diff = abs(positions[i] - current_positions[i])

                if diff > max_allowed_change:
                    self.get_logger().warn(
                        f"IK rejected: joint{i + 1} changes too much: {diff:.2f} rad"
                    )
                    return None

        except KeyError as e:
            self.get_logger().warn(f"IK solution missing joint: {e}")
            return None

        self.get_logger().info(f"IK solution accepted: {positions}")
        return positions

    def send_arm_goal(self, positions, duration=10):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = 0

        goal_msg.trajectory.points.append(point)

        goal_msg.goal_time_tolerance.sec = 10
        goal_msg.goal_time_tolerance.nanosec = 0

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

        if not self.wait_future(result_future, duration + 20):
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

    def call_gripper_empty_service(self, client, name, timeout=5.0):
        req = XArmCall.Request()
        future = client.call_async(req)

        if not self.wait_future(future, timeout):
            self.get_logger().warn(f"{name} service timeout")
            return False

        result = future.result()

        if result.ret != 0:
            self.get_logger().warn(
                f"{name} failed: ret={result.ret}, message={result.message}"
            )
            return False

        self.get_logger().info(f"{name} OK")
        return True

    def call_gripper_int_service(self, client, value, name, timeout=5.0):
        req = SetInt16.Request()
        req.data = int(value)

        future = client.call_async(req)

        if not self.wait_future(future, timeout):
            self.get_logger().warn(f"{name} service timeout")
            return False

        result = future.result()

        if result.ret != 0:
            self.get_logger().warn(
                f"{name} failed: ret={result.ret}, message={result.message}"
            )
            return False

        self.get_logger().info(f"{name} OK")
        return True

    def setup_gripper(self):
        if self.gripper_ready:
            return True

        self.get_logger().info("Setting up gripper...")

        if not self.call_gripper_empty_service(
            self.gripper_clean_client,
            "clean_gripper_error"
        ):
            return False

        if not self.call_gripper_int_service(
            self.gripper_enable_client,
            1,
            "set_gripper_enable"
        ):
            return False

        if not self.call_gripper_int_service(
            self.gripper_mode_client,
            0,
            "set_gripper_mode"
        ):
            return False

        self.gripper_ready = True
        self.get_logger().info("Gripper setup complete")
        return True

    def send_gripper_goal(self, position):
        if not self.setup_gripper():
            self.get_logger().warn("Gripper setup failed")
            return False

        req = GripperMove.Request()
        req.pos = float(position)
        req.wait = True
        req.timeout = 10.0

        self.get_logger().info(f"Sending gripper position: {position}")

        future = self.gripper_position_client.call_async(req)

        if not self.wait_future(future, 12.0):
            self.get_logger().warn("Gripper service timeout")
            return False

        result = future.result()

        if result.ret != 0:
            self.get_logger().warn(
                f"Gripper move failed: ret={result.ret}, message={result.message}"
            )
            return False

        self.get_logger().info("Gripper motion complete")
        time.sleep(1.0)
        return True

    def open_gripper(self):
        self.get_logger().info("Opening gripper to 550")
        return self.send_gripper_goal(self.GRIPPER_OPEN)

    def close_gripper(self):
        self.get_logger().info("Closing gripper to 490")
        return self.send_gripper_goal(self.GRIPPER_HOLD)

    def move_to_pose(self, x, y, z, duration=10):
        joints = self.compute_ik(x, y, z)

        if joints is None:
            return False

        self.get_logger().info(
            f"Moving to pose X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        return self.send_arm_goal(joints, duration=duration)

        if joints is None:
            return False

        self.get_logger().info(
            f"Moving to pose X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        return self.send_arm_goal(joints, duration=duration)

    def move_to_initial_detection_pose(self):
        self.get_logger().info("Moving to initial detection pose...")

        if self.send_arm_goal(self.initial_detect_pose, duration=10):
            self.get_logger().info("Initial detection pose reached")
            self.initial_pose_done = True

            self.get_logger().info("Detection pose stabilised. Requesting vision service next.")
            
        else:
            self.get_logger().warn("Failed to move to initial detection pose")

        self.busy = False

    def request_vision_and_pick(self):
        self.get_logger().info("Requesting one-shot vision detection...")

        req = VisionDetect.Request()
        req.selected_class = self.VISION_SELECTED_CLASS
        req.target_frame = self.VISION_TARGET_FRAME
        req.warmup_frames = int(self.VISION_WARMUP_FRAMES)

        future = self.vision_client.call_async(req)

        if not self.wait_future(future, 60.0):
            self.get_logger().warn("Vision service timeout")
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        result = future.result()

        if not result.success:
            self.get_logger().warn(f"Vision detection failed: {result.message}")
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        if result.target.header.frame_id != "link_base":
            self.get_logger().warn(
                f"Vision result is in {result.target.header.frame_id}, expected link_base"
            )
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        x = result.target.point.x
        y = result.target.point.y
        z = result.target.point.z

        if x < 0.10 or x > 0.75:
            self.get_logger().warn(f"Rejected X out of range: {x:.3f}")
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        if y < -0.25 or y > 0.25:
            self.get_logger().warn(f"Rejected Y out of range: {y:.3f}")
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        if z < 0.03 or z > 0.40:
            self.get_logger().warn(f"Rejected Z out of range: {z:.3f}")
            self.busy = False
            self.done_once = False
            self.vision_requested = False
            return

        self.saved_target = (x, y, z)
        self.done_once = True

        self.get_logger().info(
            f"One-shot vision target received in link_base: X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        self.pick_coffee_sequence(x, y, z)

    def pick_coffee_sequence(self, x, y, z):
        self.get_logger().info(
            f"Starting sequence at cup X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        # Current tuning values
        approach_x = x - 0.05
        approach_z = z - 0.025
        pick_x = x + self.X_PICK_OFFSET + (abs(y) * self.X_SIDE_GAIN)
        pick_z = z - 0.015
        pick_y = (y * self.Y_SIDE_SCALE) + self.Y_PICK_OFFSET

        self.get_logger().info(
            f"Using approach X={approach_x:.3f}, Y={pick_y:.3f}, Z={approach_z:.3f}"
        )

        self.get_logger().info(
            f"Using X={pick_x:.3f}, Y={pick_y:.3f}, Z={pick_z:.3f}"
        )

        if not self.open_gripper():
            self.busy = False
            return

        # 1. Move near the cup with X offset
        if not self.move_to_pose(approach_x, y, approach_z, duration=10):
            self.get_logger().warn("Failed to move to approach pose, gripper will not close")
            self.busy = False
            return
            
        # 2. Move to final pick position
        if not self.move_to_pose(pick_x, pick_y, pick_z, duration=10):
            self.get_logger().warn("Failed to move to pick pose, gripper will not close")
            self.busy = False
            return

        # 3. Close gripper
        self.get_logger().info("Reached pick pose, now closing gripper")
        if not self.close_gripper():
            self.get_logger().warn("Failed to close gripper")
            self.busy = False
            return

        time.sleep(1.0)

        if not self.send_arm_goal(self.after_pick_pose, duration=10):
            self.busy = False
            return

        if not self.send_arm_goal(self.coffee_machine_pose, duration=10):
            self.busy = False
            return

        if not self.open_gripper():
            self.busy = False
            return

        self.get_logger().info("Cup at coffee machine. Waiting 10 seconds...")
        time.sleep(10.0)

        if not self.close_gripper():
            self.busy = False
            return

        if not self.send_arm_goal(self.after_pick_pose, duration=10):
            self.busy = False
            return

        if not self.send_arm_goal(self.consumer_place_pose, duration=10):
            self.busy = False
            return
            
        if not self.open_gripper():
            self.busy = False
            return
        
        if not self.send_arm_goal(self.after_place_pose, duration=10):
            self.busy = False
            return  
        
        if not self.send_arm_goal(self.initial_pose, duration=10):
            self.busy = False
            return    
   
        # Go directly back to the original received coordinate
        #if not self.move_to_pose(x, y, z, duration=10):
         #   self.busy = False
           # return

        if not self.open_gripper():
            self.busy = False
            return

        time.sleep(1.0)

        #self.send_arm_goal(self.initial_detect_pose, duration=15)

        self.get_logger().info("Full coffee cup sequence complete")
        self.busy = False


def main(args=None):
    rclpy.init(args=args)

    node = MoveItPickPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
