import math
import time
from collections import deque
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class GazeboCupMarkerNode(Node):

    def __init__(self):
        super().__init__('gazebo_cup_marker_node')

        self.marker_name = "detected_cup_marker"

        self.spawned = False
        self.spawn_in_progress = False
        self.delete_in_progress = False

        self.spawned_position_world = None

        self.pending_respawn = False
        self.pending_target_world = None

        # Store recent valid samples to reduce YOLO/depth jitter
        self.samples_world = deque(maxlen=15)
        self.min_samples_for_stable = 8

        self.last_valid_target_time = None

        # Coordinates must remain stable before first marker spawn
        self.stable_start_time = None
        self.required_stable_time_sec = 3.0

        # Allowed coordinate jitter before considered unstable
        self.stable_xy_tolerance = 0.025   # 2.5 cm
        self.stable_z_tolerance = 0.040    # 4 cm

        self.last_stability_log_time = 0.0

        # Delete marker only if no valid target for this long
        self.target_timeout_sec = 8.0

        # Only update marker if cup position changes clearly in XY
        self.respawn_distance_xy = 0.12

        # Need repeated confirmation before updating marker
        self.required_move_confirmations = 4
        self.move_confirm_count = 0

        self.target_frame = "world"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            PointStamped,
            '/target_base',
            self.target_callback,
            10
        )

        self.spawn_client = self.create_client(
            SpawnEntity,
            '/spawn_entity'
        )

        self.delete_client = self.create_client(
            DeleteEntity,
            '/delete_entity'
        )

        self.get_logger().info("Waiting for /spawn_entity...")
        self.spawn_client.wait_for_service()

        self.get_logger().info("Waiting for /delete_entity...")
        self.delete_client.wait_for_service()

        self.create_timer(0.5, self.timer_callback)

        self.get_logger().info("Gazebo cup marker node started")
        self.get_logger().info(
            f"Marker will spawn only after coordinates are stable for "
            f"{self.required_stable_time_sec:.1f}s"
        )

    def target_callback(self, msg):
        source_frame = msg.header.frame_id

        if source_frame == "":
            source_frame = "link_base"
            msg.header.frame_id = source_frame

        x_base = msg.point.x
        y_base = msg.point.y
        z_base = msg.point.z

        if not self.valid_target_base(x_base, y_base, z_base):
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )

            target_world_msg = do_transform_point(msg, transform)

        except Exception as e:
            self.get_logger().warn(
                f"Could not transform target from {source_frame} to world: {e}"
            )
            return

        xw = target_world_msg.point.x
        yw = target_world_msg.point.y
        zw = target_world_msg.point.z

        self.samples_world.append((xw, yw, zw))
        self.last_valid_target_time = time.monotonic()

    def timer_callback(self):
        if self.last_valid_target_time is None:
            return

        age = time.monotonic() - self.last_valid_target_time

        if age > self.target_timeout_sec:
            if self.spawned and not self.delete_in_progress:
                self.get_logger().warn(
                    f"No valid /target_base for {age:.1f}s. Deleting marker."
                )
                self.pending_respawn = False
                self.pending_target_world = None
                self.move_confirm_count = 0
                self.delete_marker()

            if not self.spawned:
                self.stable_start_time = None
                self.samples_world.clear()

            return

        stable_target = self.get_stable_target_world()

        if stable_target is None:
            return

        x, y, z = stable_target

        # Before first spawn: require stable coordinates
        if not self.spawned and not self.spawn_in_progress:
            if not self.coordinates_are_stable():
                self.stable_start_time = None

                now = time.monotonic()
                if now - self.last_stability_log_time > 1.0:
                    self.get_logger().info(
                        "Waiting for coordinates to become stable before spawning marker..."
                    )
                    self.last_stability_log_time = now

                return

            now = time.monotonic()

            if self.stable_start_time is None:
                self.stable_start_time = now
                self.get_logger().info(
                    f"Coordinates stable. Holding for "
                    f"{self.required_stable_time_sec:.1f}s before spawn..."
                )
                return

            stable_duration = now - self.stable_start_time

            if stable_duration < self.required_stable_time_sec:
                if now - self.last_stability_log_time > 1.0:
                    remaining = self.required_stable_time_sec - stable_duration
                    self.get_logger().info(
                        f"Coordinates still stable. Spawning in {remaining:.1f}s..."
                    )
                    self.last_stability_log_time = now
                return

            self.get_logger().info(
                "Coordinates confirmed stable. Spawning marker now."
            )
            self.spawn_marker(x, y, z)
            return

        # After spawned: only update if cup clearly moved
        if self.spawned and self.spawned_position_world is not None:
            dist_xy = self.distance_xy(
                self.spawned_position_world,
                stable_target
            )

            if dist_xy > self.respawn_distance_xy:
                self.move_confirm_count += 1

                self.get_logger().info(
                    f"Possible cup movement detected: {dist_xy:.3f} m "
                    f"({self.move_confirm_count}/{self.required_move_confirmations})"
                )

                if (
                    self.move_confirm_count >= self.required_move_confirmations
                    and not self.delete_in_progress
                    and not self.spawn_in_progress
                ):
                    self.get_logger().info(
                        f"Cup movement confirmed: {dist_xy:.3f} m. Updating marker."
                    )

                    self.pending_respawn = True
                    self.pending_target_world = stable_target
                    self.move_confirm_count = 0

                    self.delete_marker()
            else:
                self.move_confirm_count = 0

    def coordinates_are_stable(self):
        if len(self.samples_world) < self.min_samples_for_stable:
            return False

        xs = [p[0] for p in self.samples_world]
        ys = [p[1] for p in self.samples_world]
        zs = [p[2] for p in self.samples_world]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)

        if x_range > self.stable_xy_tolerance:
            return False

        if y_range > self.stable_xy_tolerance:
            return False

        if z_range > self.stable_z_tolerance:
            return False

        return True

    def valid_target_base(self, x, y, z):
        if x < 0.10 or x > 0.85:
            self.get_logger().warn(f"Rejected X: {x:.3f}")
            return False

        if y < -0.45 or y > 0.45:
            self.get_logger().warn(f"Rejected Y: {y:.3f}")
            return False

        if z < 0.00 or z > 0.70:
            self.get_logger().warn(f"Rejected Z: {z:.3f}")
            return False

        return True

    def get_stable_target_world(self):
        if len(self.samples_world) < self.min_samples_for_stable:
            return None

        xs = [p[0] for p in self.samples_world]
        ys = [p[1] for p in self.samples_world]
        zs = [p[2] for p in self.samples_world]

        return (
            median(xs),
            median(ys),
            median(zs)
        )

    def distance_xy(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def spawn_marker(self, x, y, z):
        self.spawn_in_progress = True

        cup_sdf = """
        <sdf version="1.6">
          <model name="detected_cup_marker">
            <static>true</static>
            <link name="link">
              <visual name="visual">
                <geometry>
                  <cylinder>
                    <radius>0.04</radius>
                    <length>0.10</length>
                  </cylinder>
                </geometry>
                <material>
                  <ambient>0 1 0 1</ambient>
                  <diffuse>0 1 0 1</diffuse>
                </material>
              </visual>
            </link>
          </model>
        </sdf>
        """

        req = SpawnEntity.Request()
        req.name = self.marker_name
        req.xml = cup_sdf
        req.robot_namespace = ""
        req.reference_frame = "world"

        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.initial_pose.orientation.w = 1.0

        self.get_logger().info(
            f"Spawning marker in world at X={x:.3f}, Y={y:.3f}, Z={z:.3f}"
        )

        future = self.spawn_client.call_async(req)
        future.add_done_callback(
            lambda future: self.spawn_done_callback(future, x, y, z)
        )

    def spawn_done_callback(self, future, x, y, z):
        self.spawn_in_progress = False

        try:
            result = future.result()

            if result.success:
                self.spawned = True
                self.spawned_position_world = (x, y, z)
                self.get_logger().info("Cup marker spawned successfully")
            else:
                msg = result.status_message.lower()

                if "already" in msg:
                    self.spawned = True
                    self.spawned_position_world = (x, y, z)
                    self.get_logger().warn(
                        "Marker already exists. Treating as spawned."
                    )
                else:
                    self.spawned = False
                    self.spawned_position_world = None
                    self.get_logger().warn(
                        f"Spawn failed: {result.status_message}"
                    )

        except Exception as e:
            self.spawned = False
            self.spawned_position_world = None
            self.get_logger().warn(f"Spawn exception: {e}")

    def delete_marker(self):
        self.delete_in_progress = True

        req = DeleteEntity.Request()
        req.name = self.marker_name

        self.get_logger().info("Deleting cup marker")

        future = self.delete_client.call_async(req)
        future.add_done_callback(self.delete_done_callback)

    def delete_done_callback(self, future):
        self.delete_in_progress = False
        self.spawned = False
        self.spawned_position_world = None

        try:
            result = future.result()
            self.get_logger().info(
                f"Delete result: {result.status_message}"
            )

        except Exception as e:
            self.get_logger().warn(f"Delete exception: {e}")

        if self.pending_respawn and self.pending_target_world is not None:
            x, y, z = self.pending_target_world

            self.pending_respawn = False
            self.pending_target_world = None

            self.get_logger().info("Respawning marker at updated stable cup position")
            self.spawn_marker(x, y, z)


def main(args=None):
    rclpy.init(args=args)

    node = GazeboCupMarkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
