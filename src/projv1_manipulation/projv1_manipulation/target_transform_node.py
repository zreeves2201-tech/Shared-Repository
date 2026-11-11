import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class TargetTransformNode(Node):
    def __init__(self):
        super().__init__('target_transform_node')

        self.target_frame = 'link_base'

        # EMA filtering in the global frame to prevent drift during movement
        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None
        self.alpha = 0.2  # Lower value for more stability

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            PointStamped,
            '/detected_target',
            self.target_callback,
            10
        )

        self.pub = self.create_publisher(
            PointStamped,
            '/target_base',
            10
        )

        self.get_logger().info("Target transform node started")
        self.get_logger().info("Transforming /detected_target to link_base")

    def target_callback(self, msg):
        source_frame = msg.header.frame_id

        if source_frame == "":
            self.get_logger().warn("Rejected target: empty frame_id")
            return

        try:
            # Attempt lookup with the specific timestamp for maximum accuracy
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    msg.header.stamp,
                    timeout=Duration(seconds=0.1)
                )
            except Exception:
                # Fallback to the latest available transform if timestamped lookup fails
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5)
                )

            target_base = do_transform_point(msg, transform)
            target_base.header.stamp = self.get_clock().now().to_msg()
            target_base.header.frame_id = self.target_frame

            # Smooth the coordinate ONLY after it has been transformed to base frame
            if self.filtered_x is None:
                self.filtered_x = target_base.point.x
                self.filtered_y = target_base.point.y
                self.filtered_z = target_base.point.z
            else:
                self.filtered_x = self.alpha * target_base.point.x + (1 - self.alpha) * self.filtered_x
                self.filtered_y = self.alpha * target_base.point.y + (1 - self.alpha) * self.filtered_y
                self.filtered_z = self.alpha * target_base.point.z + (1 - self.alpha) * self.filtered_z

            target_base.point.x = self.filtered_x
            target_base.point.y = self.filtered_y
            target_base.point.z = self.filtered_z

            self.pub.publish(target_base)

            self.get_logger().info(
                f"Target in {self.target_frame}: "
                f"X={target_base.point.x:.3f}, "
                f"Y={target_base.point.y:.3f}, "
                f"Z={target_base.point.z:.3f}"
            )

        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed from {source_frame} to {self.target_frame}: {e}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = TargetTransformNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
