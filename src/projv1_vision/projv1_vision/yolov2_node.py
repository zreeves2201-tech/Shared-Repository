import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        if int(np.__version__.split('.')[0]) >= 2:
            self.get_logger().error(
                f"Detected NumPy {np.__version__}. "
                "This version is incompatible with cv_bridge on ROS Humble. "
                "Please run: pip install 'numpy<2.0'"
            )

        self.bridge = CvBridge()

        self.model = YOLO(
            "/home/user/projv1/models/best.pt"
        )

        self.person_model = YOLO(
            "yolov8n.pt"
        )

        self.depth_image = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.selected_class = None

        self.publisher_ = self.create_publisher(
            PointStamped,
            '/detected_target',
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        self.create_subscription(
            String,
            '/selected_class',
            self.selected_class_callback,
            10
        )

        self.get_logger().info(
            f"Model classes: {self.model.names}"
        )

        self.get_logger().info(
            "YOLO ROS Camera XYZ Display Node Started"
        )

    def selected_class_callback(self, msg):
        class_name = msg.data.strip()

        if class_name in ["Browncup", "Whitecup"]:
            self.selected_class = class_name

            self.get_logger().info(
                f"Selected class updated: {self.selected_class}"
            )

        else:
            self.get_logger().warn(
                f"Invalid class selected: {class_name}"
            )

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough"
        )

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def censor_people(self, image):
        results = self.person_model(
            image,
            conf=0.4,
            classes=[0]
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1
            )

        return image

    def depth_to_meters(self, depth_raw):
        if self.depth_image.dtype == np.uint16:
            return float(depth_raw) / 1000.0

        return float(depth_raw)

    def draw_xyz_text(self, image, lines, x, y):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 24

        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (x, y + i * line_height),
                font,
                font_scale,
                (0, 255, 0),
                thickness
            )

    def image_callback(self, msg):
        if any(v is None for v in [
            self.depth_image,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.selected_class
        ]):
            return

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            "bgr8"
        )

        results = self.model(
            frame,
            conf=0.4
        )

        annotated = frame.copy()

        annotated = self.censor_people(
            annotated
        )

        found_target = False

        for box in results[0].boxes:
            detected_class_id = int(box.cls[0])
            detected_class_name = self.model.names[detected_class_id]

            if detected_class_name != self.selected_class:
                continue

            found_target = True

            x1, y1, x2, y2 = box.xyxy[0]

            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)

            cv2.rectangle(
                annotated,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),
                2
            )

            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            if (
                v < 5 or
                v >= self.depth_image.shape[0] - 5 or
                u < 5 or
                u >= self.depth_image.shape[1] - 5
            ):
                self.get_logger().warn(
                    "Target centre is too close to image boundary"
                )
                continue

            depth_crop = self.depth_image[
                v - 5:v + 6,
                u - 5:u + 6
            ]

            valid_depths = depth_crop[
                np.isfinite(depth_crop) & (depth_crop > 0)
            ]

            if valid_depths.size == 0:
                self.get_logger().warn("No valid depth")
                continue

            depth_raw = np.median(valid_depths)
            depth_m = self.depth_to_meters(depth_raw)

            if depth_m <= 0.0 or depth_m > 5.0:
                self.get_logger().warn(
                    f"Rejected unrealistic depth: {depth_m:.3f} m"
                )
                continue

            # Optical camera coordinates from RealSense projection
            optical_x = (u - self.cx) * depth_m / self.fx
            optical_y = (v - self.cy) * depth_m / self.fy
            optical_z = depth_m

            # Convert optical XYZ to ROS camera-frame XYZ
            # ROS camera frame:
            # X = forward
            # Y = left
            # Z = up
            ros_x = optical_z
            ros_y = -optical_x
            ros_z = -optical_y

            target = PointStamped()
            target.header.stamp = msg.header.stamp
            target.header.frame_id = "camera_color_frame"

            target.point.x = ros_x
            target.point.y = ros_y
            target.point.z = ros_z

            self.publisher_.publish(target)

            self.get_logger().info(
                f"{self.selected_class} ROS Camera XYZ: "
                f"X={ros_x:.3f}, Y={ros_y:.3f}, Z={ros_z:.3f}"
            )

            cv2.circle(
                annotated,
                (u, v),
                5,
                (0, 255, 0),
                -1
            )

            text_lines = [
                f"{self.selected_class}",
                f"ROS Cam X: {ros_x:.3f} m",
                f"ROS Cam Y: {ros_y:.3f} m",
                f"ROS Cam Z: {ros_z:.3f} m"
            ]

            text_x = x1_i
            text_y = max(25, y1_i - 100)

            self.draw_xyz_text(
                annotated,
                text_lines,
                text_x,
                text_y
            )

        if not found_target:
            cv2.putText(
                annotated,
                f"Selected: {self.selected_class} | No target",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow(
            "YOLO ROS Camera XYZ Detection",
            annotated
        )

        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()