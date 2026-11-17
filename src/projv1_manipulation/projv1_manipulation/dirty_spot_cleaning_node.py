import math
import time

import rclpy
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPositionIK
from projv1_interfaces.srv import ActivateDirtySpotCleaning, VisionDetect
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class DirtySpotCleaningNode(Node):
    """Vision-guided urinal cleaning using the physical 33 cm brush tip.

    Important geometry for this robot:
      * MoveIt IK controls ``link_tcp`` (the existing gripper TCP).
      * The cleaning brush begins AFTER that gripper TCP and extends 0.33 m
        along the local +Z axis of ``link_tcp``.
      * Vision returns the dirty SURFACE point in ``link_base``.
      * Therefore ``link_tcp`` must stop 0.33 m behind that surface point
        along the current tool axis; it must NOT be commanded directly to the
        dirty XYZ.

    The node keeps the previously tested central entry, then moves to an
    10 mm brush-tip standoff, a 15 mm near-contact standoff, and finally
    geometric contact.  Contact compression defaults to zero for the first
    physical test.  This is geometric contact only, not force-controlled.
    """

    def __init__(self):
        super().__init__('dirty_spot_cleaning_node')

        self.callback_group = ReentrantCallbackGroup()
        self.busy = False
        self.latest_joint_state = None
        self.ready_logged = False

        # ------------------------------------------------------------------
        # Taught urinal pose / proven safe route.
        # ------------------------------------------------------------------
        self.CENTRE_POSE_CONFIGURED = True
        self.URINAL_CENTRE_POSE_DEG = [
            -5.0,
            -52.0,
            3.0,
            43.0,
            4.0,
            56.0,
            0.0,
        ]
        self.urinal_centre_pose = self.deg_list(self.URINAL_CENTRE_POSE_DEG)

        self.initial_pose = self.deg_list([
            0.0,
            -37.0,
            -3.0,
            83.0,
            0.0,
            96.0,
            -5.0,
        ])

        # ------------------------------------------------------------------
        # Motion timing.
        # ------------------------------------------------------------------
        self.CAMERA_POSE_VALIDATION_ONLY = False
        self.INITIAL_DURATION = 8.0
        self.APPROACH_DURATION = 4.5
        self.RETRACT_DURATION = 4.0
        self.Y_ALIGNMENT_DURATION = 4.0
        self.Z_STEP_DURATION = 4.0
        self.FORWARD_ENTRY_DURATION = 1.0
        self.CONTACT_STEP_DURATION = 0.8

        # ------------------------------------------------------------------
        # Geometric route and contact tuning.
        # ------------------------------------------------------------------
        self.FORWARD_ENTRY_DISTANCE = 0.12
        self.FORWARD_ENTRY_STEP = 0.06
        self.Z_ALIGNMENT_STEP = 0.01

        # Physical tool model.  The 33 cm brush starts AFTER the existing
        # gripper TCP (link_tcp), not at link_eef.  It follows link_tcp local +Z.
        self.BRUSH_EXTENSION_FROM_TCP = 0.33

        # First physical contact test: zero intentional compression.  After
        # confirming that the bristles just reach the surface, 0.002 gives
        # approximately 2 mm compression ALONG THE BRUSH AXIS.
        self.CONTACT_COMPRESSION = 0.000
        self.CLEANING_CONTACT_STANDOFF = 0.001

        # ------------------------------------------------------------------
        # TEMPORARY PHYSICAL VALIDATION MODE
        # ------------------------------------------------------------------
        # True  = temporary standoff-only validation mode.
        # False = enable the supervised first-contact cleaning sequence.
        #
        # Current commissioning route: physical brush tip goes to 20 mm, then
        # 10 mm, then zero-compression geometric contact.
        #
        # Legacy note: move only until the physical brush tip is 10 mm from the
        #         detected dirt, hold there briefly, then retract safely.
        #         No 15 mm approach, no contact, no wiping, no retry.
        # False = enable contact + short wipe + verification.
        #
        # The 10 mm standoff has now been physically validated, so this is False.
        self.STANDOFF_VALIDATION_ONLY = False
        self.STANDOFF_VALIDATION_HOLD_SEC = 5.0

        # Approach distances are measured from the PHYSICAL BRUSH TIP to the
        # detected surface, along the brush axis.
        self.FAR_STANDOFF_DISTANCE = 0.020
        self.NEAR_STANDOFF_DISTANCE = 0.010
        self.CARTESIAN_APPROACH_STEP = 0.030
        self.CONTACT_STEP = 0.001
        self.MAX_CONTACT_ADVANCE_FROM_ENTRY = 0.18
        self.MIN_ENTRY_TO_CONTACT_CLEARANCE = 0.005

        # Safety limit applies to the commanded gripper TCP, not to the low
        # dirty-surface Z.  For the sample measured in this project the dirt
        # was Z~0.131 m, but the required link_tcp contact pose is Z~0.387 m.
        self.MIN_TCP_Z = 0.29

        # Wiping motion while the brush tip is at contact X.
        self.CLEANING_Y_OFFSET = 0.0075
        self.CONTINUOUS_CLEANING_CYCLES = 2
        self.CONTINUOUS_CLEANING_SEGMENT_TIME = 1.5

        # ------------------------------------------------------------------
        # Closed-loop cleaning verification.
        # ------------------------------------------------------------------
        self.MAX_CLEANING_ATTEMPTS = 2
        self.CLEANING_REDUCTION_PASS_PERCENT = 50.0
        self.VERIFICATION_WARMUP_FRAMES = 5

        # Prevent sudden IK branch changes.
        self.MAX_JOINT_DELTA_DEG = 75.0

        # Reachability limits for dirty targets in link_base.
        self.MIN_X = 0.40
        self.MAX_X = 1.10
        self.MIN_Y = -0.30
        self.MAX_Y = 0.30
        self.MIN_Z = -0.10
        self.MAX_Z = 0.50
        self.REACH_BUFFER = 0.02

        self.SAFE_MIN_X = self.MIN_X + self.REACH_BUFFER
        self.SAFE_MAX_X = self.MAX_X - self.REACH_BUFFER
        self.SAFE_MIN_Y = self.MIN_Y + self.REACH_BUFFER
        self.SAFE_MAX_Y = self.MAX_Y - self.REACH_BUFFER
        self.SAFE_MIN_Z = self.MIN_Z + self.REACH_BUFFER
        self.SAFE_MAX_Z = self.MAX_Z - self.REACH_BUFFER

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
        ]

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group,
        )

        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/xarm7_traj_controller/follow_joint_trajectory',
            callback_group=self.callback_group,
        )

        self.vision_client = self.create_client(
            VisionDetect,
            '/visionv1/detect',
            callback_group=self.callback_group,
        )

        self.activate_service = self.create_service(
            ActivateDirtySpotCleaning,
            '/activate_dirty_spot_cleaning',
            self.activate_callback,
            callback_group=self.callback_group,
        )

        self.create_timer(1.0, self.status_timer, callback_group=self.callback_group)

        self.get_logger().info('Waiting for /compute_ik...')
        self.ik_client.wait_for_service()
        self.get_logger().info('Waiting for arm trajectory action server...')
        self.arm_action_client.wait_for_server()
        self.get_logger().info('Waiting for /visionv1/detect...')
        self.vision_client.wait_for_service()
        self.get_logger().info(
            'Dirty-spot cleaning node ready: contact + verification + retry enabled.'
        )
        self.get_logger().warn(
            'SAFE TRAJECTORY START MODE: sampled /joint_states start waypoint removed.'
        )
        self.get_logger().warn(
            f'CONTACT CLEANING ENABLED: safe trajectory-start fix retained; '
            f'{self.CLEANING_CONTACT_STANDOFF * 1000.0:.0f} mm brush-tip standoff active.'
        )
        self.get_logger().info(
            'FAST APPROACH MODE: 60 mm central-entry steps, 30 mm free-space '
            'Cartesian steps, then slow 10 mm -> contact approach.'
        )
        self.get_logger().info(
            f'CLEANING MOTION: Y scrub at {self.CLEANING_CONTACT_STANDOFF * 1000.0:.0f} mm standoff, '
            f'+/-{self.CLEANING_Y_OFFSET * 1000.0:.1f} mm Y, '
            f'{self.CONTINUOUS_CLEANING_CYCLES} cycle(s) per pass, '
            f'up to {self.MAX_CLEANING_ATTEMPTS} pass(es), '
            f'{self.CLEANING_REDUCTION_PASS_PERCENT:.0f}% reduction required, zero compression.'
        )

    @staticmethod
    def deg_list(values):
        return [math.radians(value) for value in values]

    def joint_state_callback(self, msg):
        self.latest_joint_state = msg

    def status_timer(self):
        if self.latest_joint_state is None:
            self.get_logger().warn('Still waiting for /joint_states...')
            return
        if not self.ready_logged:
            self.get_logger().info('Ready. Call /activate_dirty_spot_cleaning to begin.')
            self.ready_logged = True

    @staticmethod
    def reset_response(response):
        response.success = False
        response.message = ''
        response.adjustment_required = False
        response.diff_x = 0.0
        response.diff_y = 0.0
        response.diff_z = 0.0

    def wait_future(self, future, timeout_sec):
        start_time = time.time()
        while rclpy.ok():
            if future.done():
                return True
            if time.time() - start_time > timeout_sec:
                return False
            time.sleep(0.05)
        return False

    def activate_callback(self, request, response):
        self.reset_response(response)

        if self.latest_joint_state is None:
            response.message = 'Cannot start: /joint_states is not available.'
            return response
        if self.busy:
            response.message = 'Dirty-spot cleaning sequence is already running.'
            return response

        selected_class = request.selected_class.strip()
        target_frame = request.target_frame.strip() or 'link_base'
        warmup_frames = int(request.warmup_frames)

        # Detection supports ToiletBowl, but this manipulation path has only
        # been taught and safety-tested for the urinal fixture.
        if selected_class != 'Urinal':
            response.message = (
                "Cleaning manipulation currently supports only 'Urinal'. "
                'ToiletBowl detection can still be used, but it needs its own '
                'taught approach/contact profile before automatic cleaning.'
            )
            return response

        if target_frame != 'link_base':
            response.message = (
                "Rejected: dirty-spot manipulation requires target_frame='link_base'."
            )
            return response

        if warmup_frames < 0 or warmup_frames > 100:
            response.message = 'warmup_frames must be between 0 and 100.'
            return response

        self.busy = True
        try:
            success, message, diff = self.run_sequence(
                selected_class,
                target_frame,
                warmup_frames,
            )
            response.success = success
            response.message = message
            response.diff_x, response.diff_y, response.diff_z = diff
            response.adjustment_required = self.target_needs_base_adjustment(*diff)
            return response
        except Exception as exc:
            self.get_logger().error(f'Cleaning sequence exception: {exc}')
            response.message = f'Cleaning sequence failed: {exc}'
            return response
        finally:
            self.busy = False

    def run_sequence(self, selected_class, target_frame, warmup_frames):
        """Run contact cleaning and close the loop with visual verification."""
        if not self.CENTRE_POSE_CONFIGURED or len(self.URINAL_CENTRE_POSE_DEG) != 7:
            return (
                False,
                'Urinal centre pose is not configured correctly.',
                (0.0, 0.0, 0.0),
            )

        if not self.move_to_initial_pose():
            return False, 'Failed to move to the initial safe pose.', (0.0, 0.0, 0.0)

        if self.CAMERA_POSE_VALIDATION_ONLY:
            self.get_logger().warn(
                'CAMERA-POSE-ONLY VALIDATION COMPLETE: no vision, approach, '
                'contact, or cleaning motion will be executed.'
            )
            return (
                True,
                'Camera-pose-only validation completed successfully. '
                'No vision/contact/cleaning sequence was executed.',
                (0.0, 0.0, 0.0),
            )

        self.get_logger().info('Requesting stable dirty target from vision...')
        detection = self.call_vision_service(selected_class, target_frame, warmup_frames)
        if detection is None:
            return (
                False,
                'Initial vision detection failed or timed out.',
                (0.0, 0.0, 0.0),
            )

        if not detection.fixture_found:
            return False, 'Urinal fixture was not confirmed.', (0.0, 0.0, 0.0)

        if not detection.dirty_spot_found:
            return (
                True,
                'Urinal detected and verified clean. No dirty spot was found; '
                'no cleaning motion was required.',
                (0.0, 0.0, 0.0),
            )

        baseline_area = max(1, int(detection.dirty_area_px))
        latest_area = baseline_area
        latest_reduction = 0.0

        self.get_logger().info(
            f'Initial dirty area: {baseline_area} px | '
            f'dirty confidence: {detection.dirty_confidence * 100:.1f}%'
        )

        for attempt in range(1, self.MAX_CLEANING_ATTEMPTS + 1):
            target = detection.target
            dirty_x = float(target.point.x)
            dirty_y = float(target.point.y)
            dirty_z = float(target.point.z)

            self.get_logger().info(
                f'Cleaning attempt {attempt}/{self.MAX_CLEANING_ATTEMPTS}: '
                f'X={dirty_x:.3f}, Y={dirty_y:.3f}, Z={dirty_z:.3f}, '
                f'area={int(detection.dirty_area_px)} px'
            )

            diff = self.calculate_base_diff(dirty_x, dirty_y, dirty_z)
            if self.target_needs_base_adjustment(*diff):
                return (
                    False,
                    'Dirty target is outside the safe arm range; base adjustment is required.',
                    diff,
                )

            # Do NOT compare dirty_z directly with the old TCP height limit.
            # dirty_z is the physical surface height; the commanded link_tcp
            # remains 33 cm behind the brush tip and is validated inside
            # execute_cleaning_pass().

            if not self.execute_cleaning_pass(dirty_x, dirty_y, dirty_z):
                return (
                    False,
                    f'Cleaning attempt {attempt} failed. Check the robot/controller '
                    'before sending another command.',
                    (0.0, 0.0, 0.0),
                )

            # During the staged physical validation we intentionally stop at
            # the 10 mm standoff and return home.  Do NOT run post-cleaning
            # verification/retry because no cleaning was performed.
            if self.STANDOFF_VALIDATION_ONLY:
                return (
                    True,
                    '10 mm brush-tip standoff validation completed successfully. '
                    'No surface contact or wiping was commanded. Measure the '
                    'physical brush-tip-to-dirt gap before enabling the next stage.',
                    (0.0, 0.0, 0.0),
                )

            # The camera is back at the known initial pose before verification.
            self.get_logger().info('Cleaning pass complete. Verifying fixture again...')
            verification = self.call_vision_service(
                selected_class,
                target_frame,
                self.VERIFICATION_WARMUP_FRAMES,
            )

            if verification is None:
                return (
                    False,
                    'Cleaning motion completed, but post-cleaning visual verification '
                    'failed or timed out.',
                    (0.0, 0.0, 0.0),
                )

            if not verification.fixture_found:
                return (
                    False,
                    'Cleaning motion completed, but the urinal could not be confirmed '
                    'during post-cleaning verification.',
                    (0.0, 0.0, 0.0),
                )

            if not verification.dirty_spot_found:
                self.get_logger().info(
                    'POST-CLEANING RESULT: no dirty spot detected over the confirmation window.'
                )
                return (
                    True,
                    f'Cleaning successful after {attempt} pass(es). '
                    f'Dirty area reduced from {baseline_area} px to 0 px '
                    '(100.0% reduction); no remaining dirty spot was detected.',
                    (0.0, 0.0, 0.0),
                )

            latest_area = int(verification.dirty_area_px)
            latest_reduction = self.calculate_reduction_percent(baseline_area, latest_area)
            self.get_logger().info(
                'POST-CLEANING RESULT: '
                f'before={baseline_area} px, after={latest_area} px, '
                f'reduction={latest_reduction:.1f}%'
            )

            if latest_reduction >= self.CLEANING_REDUCTION_PASS_PERCENT:
                return (
                    True,
                    f'Cleaning verified after {attempt} pass(es). '
                    f'Dirty area reduced from {baseline_area} px to {latest_area} px '
                    f'({latest_reduction:.1f}% reduction), meeting the '
                    f'{self.CLEANING_REDUCTION_PASS_PERCENT:.0f}% pass threshold.',
                    (0.0, 0.0, 0.0),
                )

            if attempt < self.MAX_CLEANING_ATTEMPTS:
                self.get_logger().warn(
                    f'Cleaning reduction is {latest_reduction:.1f}%, below the '
                    f'{self.CLEANING_REDUCTION_PASS_PERCENT:.0f}% threshold. '
                    'Using the newly detected remaining dirty target for one retry.'
                )
                detection = verification
                continue

        return (
            False,
            f'Cleaning motion completed for {self.MAX_CLEANING_ATTEMPTS} passes, '
            f'but visual verification measured only {latest_reduction:.1f}% '
            f'dirty-area reduction ({baseline_area} px -> {latest_area} px), '
            f'below the {self.CLEANING_REDUCTION_PASS_PERCENT:.0f}% pass threshold.',
            (0.0, 0.0, 0.0),
        )

    def execute_cleaning_pass(self, dirty_x, dirty_y, dirty_z):
        """Execute one contact wipe using the 33 cm brush offset from link_tcp."""
        self.get_logger().info('Moving to manually taught urinal centre pose...')
        if not self.send_arm_goal(self.urinal_centre_pose, self.APPROACH_DURATION):
            return False

        tcp_position = self.get_current_tcp_position()
        orientation = self.get_current_eef_orientation()
        brush_tip_position = self.get_frame_position('brush_tip')
        if tcp_position is None or orientation is None or brush_tip_position is None:
            self.get_logger().error(
                'Could not read link_tcp / brush_tip geometry at taught centre pose.'
            )
            self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
            return False

        centre = tuple(float(v) for v in tcp_position)
        centre_x, centre_y, centre_z = centre
        axis = self.local_z_axis_from_quaternion(orientation)

        # Verify that the TF tree really represents the physical tool we are
        # about to use.  This catches the old launch definition where
        # brush_tip was incorrectly measured from link_eef.
        measured_brush_length = self.distance3(centre, brush_tip_position)
        predicted_tip = self.add_scaled_vector(
            centre, axis, self.BRUSH_EXTENSION_FROM_TCP
        )
        tip_model_error = self.distance3(predicted_tip, brush_tip_position)

        self.get_logger().info(
            f'Taught link_tcp: X={centre_x:.3f}, Y={centre_y:.3f}, Z={centre_z:.3f}'
        )
        self.get_logger().info(
            'Brush axis in link_base: '
            f'[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]'
        )
        self.get_logger().info(
            f'brush_tip TF length from link_tcp={measured_brush_length:.3f} m; '
            f'model error={tip_model_error * 1000.0:.1f} mm'
        )

        if abs(measured_brush_length - self.BRUSH_EXTENSION_FROM_TCP) > 0.015:
            self.get_logger().error(
                'Rejected contact: brush_tip TF does not represent a 0.33 m '
                'extension from link_tcp. Replace/relaunch cleaning_full_system.launch.py '
                'before contact testing.'
            )
            self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
            return False

        if tip_model_error > 0.010:
            self.get_logger().error(
                'Rejected contact: brush_tip TF is not aligned with link_tcp local +Z.'
            )
            self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
            return False

        dirty = (float(dirty_x), float(dirty_y), float(dirty_z))
        current_tip_distance = self.distance3(brush_tip_position, dirty)

        # link_tcp pose required for a physical brush-tip target:
        #   physical_tip = link_tcp + 0.33 * axis
        # so:
        #   link_tcp = physical_tip - 0.33 * axis
        far_tcp = self.tcp_pose_for_surface(
            dirty, axis, standoff=self.FAR_STANDOFF_DISTANCE, compression=0.0
        )
        near_tcp = self.tcp_pose_for_surface(
            dirty, axis, standoff=self.NEAR_STANDOFF_DISTANCE, compression=0.0
        )
        contact_tcp = self.tcp_pose_for_surface(
            dirty, axis, standoff=0.0, compression=self.CONTACT_COMPRESSION
        )
        cleaning_tcp = self.tcp_pose_for_surface(
            dirty,
            axis,
            standoff=self.CLEANING_CONTACT_STANDOFF,
            compression=0.0,
        )

        entry = (centre_x + self.FORWARD_ENTRY_DISTANCE, centre_y, centre_z)

        self.get_logger().info(
            f'Physical brush tip is {current_tip_distance * 100.0:.1f} cm from dirt '
            'at the taught pose.'
        )
        self.get_logger().info(
            'Computed link_tcp contact target (33 cm brush offset applied): '
            f'X={contact_tcp[0]:.3f}, Y={contact_tcp[1]:.3f}, Z={contact_tcp[2]:.3f}'
        )
        self.get_logger().info(
            'Computed MICRO-SCRUB target (3 mm brush-tip standoff): '
            f'X={cleaning_tcp[0]:.3f}, Y={cleaning_tcp[1]:.3f}, Z={cleaning_tcp[2]:.3f}'
        )
        self.get_logger().info(
            'Approach plan: '
            f'entry=({entry[0]:.3f},{entry[1]:.3f},{entry[2]:.3f}) -> '
            f'20mm=({far_tcp[0]:.3f},{far_tcp[1]:.3f},{far_tcp[2]:.3f}) -> '
            f'10mm=({near_tcp[0]:.3f},{near_tcp[1]:.3f},{near_tcp[2]:.3f}) -> '
            f'contact=({contact_tcp[0]:.3f},{contact_tcp[1]:.3f},{contact_tcp[2]:.3f})'
        )

        # In validation mode, only the 20 mm standoff must pass the TCP
        # safety envelope.  The 10 mm/contact poses are intentionally NOT
        # authorised yet, so their lower Z values must not block this test.
        poses_to_validate = [('20 mm standoff', far_tcp)]
        if not self.STANDOFF_VALIDATION_ONLY:
            poses_to_validate.extend([
                ('10 mm standoff', near_tcp),
                ('micro-scrub 3 mm standoff', cleaning_tcp),
            ])

        for label, pose in poses_to_validate:
            if not self.tcp_pose_is_safe(*pose):
                self.get_logger().error(
                    f'Rejected {label}: link_tcp pose '
                    f'({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) '
                    'is outside the configured TCP safety envelope.'
                )
                self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
                return False

        # Contact-specific checks are skipped during standoff-only validation.
        if not self.STANDOFF_VALIDATION_ONLY:
            entry_to_contact = self.distance3(entry, contact_tcp)
            if entry_to_contact <= self.MIN_ENTRY_TO_CONTACT_CLEARANCE:
                self.get_logger().error(
                    'Rejected contact plan: entry is already too close to contact.'
                )
                self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
                return False

            if entry_to_contact > self.MAX_CONTACT_ADVANCE_FROM_ENTRY:
                self.get_logger().error(
                    f'Rejected contact plan: entry-to-contact distance '
                    f'{entry_to_contact:.3f} m exceeds '
                    f'{self.MAX_CONTACT_ADVANCE_FROM_ENTRY:.3f} m.'
                )
                self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)
                return False

        # 1. Fast free-space entry: same tested 12 cm route, but only two 60 mm steps.
        if not self.move_x_incrementally(
            centre_x, entry[0], centre_y, centre_z,
            self.FORWARD_ENTRY_STEP, self.FORWARD_ENTRY_DURATION,
            label='central entry',
        ):
            return False

        # 2. Faster free-space transition to the validated 20 mm standoff using 30 mm steps.
        if not self.move_linearly_incremental(
            entry, far_tcp, self.CARTESIAN_APPROACH_STEP,
            self.FORWARD_ENTRY_DURATION, label='20 mm brush-tip standoff',
        ):
            return False

        # TEMPORARY staged validation: stop at 10 mm, hold so the physical
        # gap can be measured/observed, then retract along the exact same path.
        if self.STANDOFF_VALIDATION_ONLY:
            self.get_logger().warn(
                'VALIDATION MODE: reached commanded 10 mm brush-tip standoff. '
                f'Holding for {self.STANDOFF_VALIDATION_HOLD_SEC:.1f} s. '
                'NO contact or wiping will be commanded.'
            )
            time.sleep(self.STANDOFF_VALIDATION_HOLD_SEC)

            if not self.move_linearly_incremental(
                far_tcp, entry, self.CARTESIAN_APPROACH_STEP,
                self.FORWARD_ENTRY_DURATION, label='validation retraction',
            ):
                return False

            if not self.move_x_incrementally(
                entry[0], centre_x, centre_y, centre_z,
                self.FORWARD_ENTRY_STEP, self.FORWARD_ENTRY_DURATION,
                label='central exit after validation',
            ):
                return False

            if not self.send_arm_goal(self.urinal_centre_pose, self.RETRACT_DURATION):
                return False

            return self.move_to_initial_pose()

        # 3. Approach from 20 mm to the validated 10 mm standoff.
        if not self.move_linearly_incremental(
            far_tcp, near_tcp, self.CARTESIAN_APPROACH_STEP,
            self.FORWARD_ENTRY_DURATION, label='10 mm near-contact approach',
        ):
            return False

        # 4. Final 10 mm in 1 mm increments. With CONTACT_COMPRESSION=0,
        # the physical brush tip is commanded exactly to the detected surface.
        self.get_logger().warn(
            'MICRO-SCRUB APPROACH: 10 mm -> 3 mm brush-tip standoff '
            'in 1 mm steps. Geometric contact will NOT be commanded.'
        )
        self.get_logger().warn(
            'MICRO-SCRUB TEST: +/-5 mm base-Y only, 1 slow cycle, '
            'zero intentional compression.'
        )
        time.sleep(0.5)

        if not self.move_linearly_incremental(
            near_tcp, cleaning_tcp, self.CONTACT_STEP,
            self.CONTACT_STEP_DURATION, label='micro-scrub approach',
        ):
            return False

        if not self.execute_micro_y_cleaning(
            centre=cleaning_tcp,
            y_offset=self.CLEANING_Y_OFFSET,
            cycles=self.CONTINUOUS_CLEANING_CYCLES,
        ):
            return False

        # Retract to the already validated 10 mm standoff.
        if not self.move_linearly_incremental(
            cleaning_tcp, near_tcp, self.CONTACT_STEP,
            self.CONTACT_STEP_DURATION, label='micro-scrub release',
        ):
            return False

        if not self.move_linearly_incremental(
            near_tcp, far_tcp, self.CARTESIAN_APPROACH_STEP,
            self.FORWARD_ENTRY_DURATION, label='surface retraction',
        ):
            return False

        if not self.move_linearly_incremental(
            far_tcp, entry, self.CARTESIAN_APPROACH_STEP,
            self.FORWARD_ENTRY_DURATION, label='return to central entry',
        ):
            return False

        if not self.move_x_incrementally(
            entry[0], centre_x, centre_y, centre_z,
            self.FORWARD_ENTRY_STEP, self.FORWARD_ENTRY_DURATION,
            label='central exit',
        ):
            return False

        if not self.send_arm_goal(self.urinal_centre_pose, self.RETRACT_DURATION):
            return False

        return self.move_to_initial_pose()

    def move_to_initial_pose(self):
        self.get_logger().info('Moving to initial safe / camera pose...')
        return self.send_arm_goal(self.initial_pose, self.INITIAL_DURATION)

    @staticmethod
    def calculate_reduction_percent(before_area, after_area):
        before = max(1.0, float(before_area))
        after = max(0.0, float(after_area))
        return max(0.0, min(100.0, (before - after) / before * 100.0))

    @staticmethod
    def distance3(a, b):
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )

    @staticmethod
    def add_scaled_vector(point, vector, scale):
        return (
            float(point[0]) + float(scale) * float(vector[0]),
            float(point[1]) + float(scale) * float(vector[1]),
            float(point[2]) + float(scale) * float(vector[2]),
        )

    @staticmethod
    def local_z_axis_from_quaternion(q):
        # Third column of the quaternion rotation matrix: local +Z in base.
        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)
        axis = (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
        norm = math.sqrt(sum(value * value for value in axis))
        if norm < 1e-9:
            raise ValueError('Invalid link_tcp orientation: zero tool-axis norm.')
        return tuple(value / norm for value in axis)

    def tcp_pose_for_surface(self, dirty, axis, standoff=0.0, compression=0.0):
        # Desired physical tip is standoff behind the surface, or compression
        # beyond it, measured along the brush axis.
        offset_from_surface = (
            self.BRUSH_EXTENSION_FROM_TCP + float(standoff) - float(compression)
        )
        return self.add_scaled_vector(dirty, axis, -offset_from_surface)

    def tcp_pose_is_safe(self, x, y, z):
        return (
            self.SAFE_MIN_X <= float(x) <= self.SAFE_MAX_X
            and self.SAFE_MIN_Y <= float(y) <= self.SAFE_MAX_Y
            and max(self.SAFE_MIN_Z, self.MIN_TCP_Z) <= float(z) <= self.SAFE_MAX_Z
        )

    def get_frame_position(self, frame_id):
        for attempt in range(20):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'link_base', frame_id, rclpy.time.Time()
                )
                t = transform.transform.translation
                return float(t.x), float(t.y), float(t.z)
            except Exception as exc:
                self.get_logger().warn(
                    f'Waiting for link_base -> {frame_id} position '
                    f'({attempt + 1}/20): {exc}'
                )
                time.sleep(0.2)
        return None

    def move_linearly_incremental(
        self, start, target, step_size, duration, label='Cartesian movement'
    ):
        if step_size <= 0.0:
            self.get_logger().error('Cartesian step size must be greater than zero.')
            return False

        sx, sy, sz = (float(v) for v in start)
        tx, ty, tz = (float(v) for v in target)
        dx, dy, dz = tx - sx, ty - sy, tz - sz
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= 1e-6:
            return True

        steps = max(1, int(math.ceil(distance / float(step_size))))
        for index in range(1, steps + 1):
            ratio = index / steps
            x = sx + dx * ratio
            y = sy + dy * ratio
            z = sz + dz * ratio
            self.get_logger().info(
                f'{label} {index}/{steps}: X={x:.3f}, Y={y:.3f}, Z={z:.3f}'
            )
            if not self.move_to_pose(x, y, z, duration):
                self.get_logger().error(
                    f'{label} failed at X={x:.3f}, Y={y:.3f}, Z={z:.3f}.'
                )
                return False
            time.sleep(0.15)
        return True

    def move_x_incrementally(
        self,
        start_x,
        target_x,
        y,
        z,
        step_size,
        duration,
        label='X movement',
    ):
        if step_size <= 0.0:
            self.get_logger().error('X step size must be greater than zero.')
            return False

        current_x = float(start_x)
        target_x = float(target_x)
        direction = 1.0 if target_x >= current_x else -1.0

        while rclpy.ok():
            remaining = abs(target_x - current_x)
            if remaining <= 1e-6:
                return True
            movement = min(step_size, remaining)
            next_x = current_x + direction * movement
            self.get_logger().info(
                f'{label}: X={next_x:.3f}, Y={y:.3f}, Z={z:.3f}'
            )
            if not self.move_to_pose(next_x, y, z, duration):
                self.get_logger().error(f'{label} failed at X={next_x:.3f}.')
                return False
            current_x = next_x
            time.sleep(0.15)
        return False

    def move_z_incrementally(self, x, y, start_z, target_z, step_size):
        if step_size <= 0.0:
            self.get_logger().error('Z step size must be greater than zero.')
            return False

        current_z = float(start_z)
        target_z = float(target_z)
        direction = 1.0 if target_z >= current_z else -1.0

        while rclpy.ok():
            remaining = abs(target_z - current_z)
            if remaining <= 1e-6:
                return True
            movement = min(step_size, remaining)
            next_z = current_z + direction * movement
            self.get_logger().info(
                f'Z alignment: X={x:.3f}, Y={y:.3f}, Z={next_z:.3f}'
            )
            if not self.move_to_pose(x, y, next_z, self.Z_STEP_DURATION):
                self.get_logger().error(f'Z alignment failed at Z={next_z:.3f}.')
                return False
            current_z = next_z
            time.sleep(0.2)
        return False

    def call_vision_service(self, selected_class, target_frame, warmup_frames):
        request = VisionDetect.Request()
        request.selected_class = selected_class
        request.target_frame = target_frame
        request.warmup_frames = int(warmup_frames)

        future = self.vision_client.call_async(request)
        if not self.wait_future(future, 75.0):
            self.get_logger().warn('Vision service timeout.')
            return None

        result = future.result()
        if result is None or not result.success:
            message = 'No result' if result is None else result.message
            self.get_logger().warn(f'Vision failed: {message}')
            return None

        if result.dirty_spot_found and result.target.header.frame_id != target_frame:
            self.get_logger().warn(
                f'Vision returned frame {result.target.header.frame_id}; expected {target_frame}.'
            )
            return None

        self.get_logger().info(
            f'Vision result: fixture={result.fixture_found}, dirt={result.dirty_spot_found}, '
            f'fixture_conf={result.fixture_confidence * 100:.1f}%, '
            f'dirty_conf={result.dirty_confidence * 100:.1f}%, '
            f'dirty_area={result.dirty_area_px} px | {result.message}'
        )
        return result

    def calculate_base_diff(self, x, y, z):
        if x > self.MAX_X:
            diff_x = x - self.SAFE_MAX_X
        elif x < self.MIN_X:
            diff_x = -(self.SAFE_MIN_X - x)
        else:
            diff_x = 0.0

        if y > self.MAX_Y:
            diff_y = y - self.SAFE_MAX_Y
        elif y < self.MIN_Y:
            diff_y = y - self.SAFE_MIN_Y
        else:
            diff_y = 0.0

        if z > self.MAX_Z:
            diff_z = z - self.SAFE_MAX_Z
        elif z < self.MIN_Z:
            diff_z = z - self.SAFE_MIN_Z
        else:
            diff_z = 0.0

        return diff_x, diff_y, diff_z

    @staticmethod
    def target_needs_base_adjustment(diff_x, diff_y, diff_z):
        return any(abs(value) > 1e-6 for value in (diff_x, diff_y, diff_z))

    def get_current_joint_positions(self):
        if self.latest_joint_state is None:
            return None
        joint_map = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        try:
            return [joint_map[name] for name in self.joint_names]
        except KeyError as exc:
            self.get_logger().warn(f'Missing joint state: {exc}')
            return None

    def get_current_tcp_position(self):
        for attempt in range(20):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'link_base',
                    'link_tcp',
                    rclpy.time.Time(),
                )
                t = transform.transform.translation
                return float(t.x), float(t.y), float(t.z)
            except Exception as exc:
                self.get_logger().warn(
                    f'Waiting for link_base -> link_tcp position ({attempt + 1}/20): {exc}'
                )
                time.sleep(0.2)
        return None

    def get_current_eef_orientation(self):
        for attempt in range(20):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'link_base',
                    'link_tcp',
                    rclpy.time.Time(),
                )
                return transform.transform.rotation
            except Exception as exc:
                self.get_logger().warn(
                    f'Waiting for link_base -> link_tcp TF ({attempt + 1}/20): {exc}'
                )
                time.sleep(0.2)
        return None

    def compute_ik(self, x, y, z, reference_positions=None):
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            self.get_logger().warn('No complete joint state is available.')
            return None

        orientation = self.get_current_eef_orientation()
        if orientation is None:
            self.get_logger().warn('Could not obtain end-effector orientation.')
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'link_base'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation = orientation

        request = GetPositionIK.Request()
        request.ik_request.group_name = 'xarm7'
        request.ik_request.ik_link_name = 'link_tcp'
        request.ik_request.pose_stamped = pose
        request.ik_request.timeout.sec = 5
        request.ik_request.avoid_collisions = True

        robot_state = RobotState()
        robot_state.joint_state = self.latest_joint_state
        request.ik_request.robot_state = robot_state

        future = self.ik_client.call_async(request)
        if not self.wait_future(future, 10.0):
            self.get_logger().warn('IK service timeout.')
            return None

        result = future.result()
        if result is None or result.error_code.val != 1:
            error_code = None if result is None else result.error_code.val
            self.get_logger().warn(f'IK failed with error code: {error_code}')
            return None

        joint_map = dict(
            zip(result.solution.joint_state.name, result.solution.joint_state.position)
        )
        try:
            solution = [joint_map[name] for name in self.joint_names]
        except KeyError as exc:
            self.get_logger().warn(f'IK solution missing joint: {exc}')
            return None

        reference = current_positions if reference_positions is None else reference_positions
        if not self.joint_delta_is_safe(reference, solution):
            return None

        return solution

    def joint_delta_is_safe(self, start_positions, end_positions):
        max_delta = math.radians(self.MAX_JOINT_DELTA_DEG)
        for index, (start, end) in enumerate(zip(start_positions, end_positions), start=1):
            delta = abs(self.shortest_angular_distance(start, end))
            if delta > max_delta:
                self.get_logger().warn(
                    f'Rejected joint path: joint{index} jump is '
                    f'{math.degrees(delta):.1f} degrees; '
                    f'limit is {self.MAX_JOINT_DELTA_DEG:.1f} degrees.'
                )
                return False
        return True

    @staticmethod
    def shortest_angular_distance(start, end):
        return math.atan2(math.sin(end - start), math.cos(end - start))

    def move_to_pose(self, x, y, z, duration):
        self.get_logger().info(f'IK pose: X={x:.3f}, Y={y:.3f}, Z={z:.3f}')
        joints = self.compute_ik(x, y, z)
        if joints is None:
            self.get_logger().error(
                f'No acceptable IK solution for X={x:.3f}, Y={y:.3f}, Z={z:.3f}'
            )
            return False
        return self.send_arm_goal(joints, duration)

    def execute_micro_y_cleaning(self, centre, y_offset, cycles):
        """Tiny Y-only scrub while remaining 3 mm short of geometric contact."""
        if y_offset <= 0.0 or cycles < 1:
            self.get_logger().error('Invalid micro-scrub configuration.')
            return False

        cx, cy, cz = centre
        negative_y = cy - y_offset
        positive_y = cy + y_offset

        cartesian_waypoints = []
        for _ in range(cycles):
            cartesian_waypoints.extend([
                (cx, negative_y, cz),
                (cx, positive_y, cz),
                (cx, cy, cz),
            ])

        for index, pose in enumerate(cartesian_waypoints, start=1):
            if not self.tcp_pose_is_safe(*pose):
                self.get_logger().error(
                    f'Rejected micro-scrub waypoint {index}: '
                    f'({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}) '
                    'is outside the configured TCP safety envelope.'
                )
                return False

        self.get_logger().warn(
            'MICRO Y SCRUB: '
            f'centre=({cx:.3f},{cy:.3f},{cz:.3f}), '
            f'Y=+/-{y_offset * 1000.0:.0f} mm, '
            f'cycles={cycles}, '
            f'brush-tip standoff={self.CLEANING_CONTACT_STANDOFF * 1000.0:.0f} mm'
        )

        joint_waypoints = []
        reference = self.get_current_joint_positions()
        if reference is None:
            return False

        for index, (target_x, target_y, target_z) in enumerate(
            cartesian_waypoints, start=1
        ):
            joints = self.compute_ik(
                target_x,
                target_y,
                target_z,
                reference_positions=reference,
            )
            if joints is None:
                self.get_logger().error(
                    f'Could not compute safe micro-scrub IK waypoint {index}.'
                )
                return False
            joint_waypoints.append(joints)
            reference = joints

        return self.send_multi_point_arm_goal(
            joint_waypoints,
            seconds_per_segment=self.CONTINUOUS_CLEANING_SEGMENT_TIME,
        )

    def send_multi_point_arm_goal(self, joint_waypoints, seconds_per_segment=1.0):
        if not joint_waypoints or seconds_per_segment <= 0.0:
            return False

        current = self.get_current_joint_positions()
        if current is None:
            return False

        previous = current
        for waypoint in joint_waypoints:
            if not self.joint_delta_is_safe(previous, waypoint):
                self.get_logger().error(
                    'Cleaning trajectory rejected because consecutive IK waypoints '
                    'change configuration too sharply.'
                )
                return False
            previous = waypoint

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        # Do not command a sampled /joint_states value as a start waypoint.
        # The trajectory controller uses its actual controller state.

        for index, positions in enumerate(joint_waypoints, start=1):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.velocities = [0.0] * len(self.joint_names)
            total_time = index * seconds_per_segment
            whole_seconds = int(total_time)
            point.time_from_start.sec = whole_seconds
            point.time_from_start.nanosec = int(
                (total_time - whole_seconds) * 1_000_000_000
            )
            goal.trajectory.points.append(point)

        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(seconds=0.2)
        ).to_msg()
        goal.goal_time_tolerance.sec = 10

        total_duration = len(joint_waypoints) * seconds_per_segment
        future = self.arm_action_client.send_goal_async(goal)
        if not self.wait_future(future, 5.0):
            self.get_logger().error('Continuous cleaning goal-send timeout.')
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Continuous cleaning trajectory was rejected.')
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, total_duration + 20.0):
            self.get_logger().error('Continuous cleaning result timeout.')
            return False

        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result.error_code != 0:
            code = None if wrapped_result is None else wrapped_result.result.error_code
            self.get_logger().error(f'Continuous cleaning trajectory failed: {code}')
            return False

        time.sleep(0.3)
        return True


    def send_arm_goal(self, positions, duration):
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            self.get_logger().error(
                'Cannot send arm goal: joint positions unavailable.'
            )
            return False

        if len(positions) != len(self.joint_names):
            self.get_logger().error(
                'Cannot send arm goal: wrong number of joints.'
            )
            return False

        # /joint_states is used only for the existing safety comparison.
        # It is NOT inserted into the commanded trajectory.
        if not self.joint_delta_is_safe(current_positions, positions):
            self.get_logger().error(
                'Arm goal rejected: target joint delta exceeds configured safety limit.'
            )
            return False

        duration = max(float(duration), 2.0)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        target_point = JointTrajectoryPoint()
        target_point.positions = list(positions)
        target_point.velocities = [0.0] * len(self.joint_names)
        target_point.time_from_start.sec = int(duration)
        target_point.time_from_start.nanosec = int(
            (duration - int(duration)) * 1_000_000_000
        )

        # Let JointTrajectoryController interpolate from its actual controller
        # state directly to the requested target.
        goal.trajectory.points = [target_point]
        goal.trajectory.header.stamp = (
            self.get_clock().now() + Duration(seconds=0.2)
        ).to_msg()
        goal.goal_time_tolerance.sec = 10

        future = self.arm_action_client.send_goal_async(goal)
        if not self.wait_future(future, 15.0):
            self.get_logger().error('Arm goal response timeout.')
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Arm goal was rejected by the controller.')
            return False

        result_future = goal_handle.get_result_async()
        if not self.wait_future(result_future, duration + 20.0):
            self.get_logger().error('Arm result timeout.')
            return False

        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error('Arm action returned no result.')
            return False

        result = wrapped_result.result
        if result.error_code != 0:
            self.get_logger().error(
                f'Arm trajectory failed. Error code: {result.error_code}; '
                f'error string: {result.error_string!r}'
            )
            return False

        time.sleep(0.3)
        return True

def main(args=None):
    rclpy.init(args=args)
    node = DirtySpotCleaningNode()
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
