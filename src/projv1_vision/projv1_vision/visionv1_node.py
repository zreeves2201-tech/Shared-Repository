import time
import os
import contextlib
os.environ["YOLO_VERBOSE"] = "False"
import threading

import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

from cv_bridge import CvBridge

import cv2
import numpy as np

from ultralytics import YOLO

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from projv1_interfaces.srv import VisionDetect


class VisionV1Node(Node):
    """
    VisionV1 node:
    - Continuous YOLO preview for /visionv1/annotated_image
    - One-shot official robot detection only when /visionv1/detect service is called

    Continuous preview does NOT continuously publish /detected_target or /target_base.
    Those are published only once per valid service request.
    """

    def __init__(self):
        super().__init__('visionv1_node')
        self.shutting_down = False

        # Silence normal logs. Only service result will print once.
        self.get_logger().set_level(LoggingSeverity.FATAL)

        if int(np.__version__.split('.')[0]) >= 2:
            self.get_logger().error(
                f"Detected NumPy {np.__version__}. "
                "This version is incompatible with cv_bridge on ROS Humble. "
                "Please run: pip install 'numpy<2.0'"
            )

        self.callback_group = ReentrantCallbackGroup()
        self.bridge = CvBridge()

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.model = YOLO("/home/user/projv1/models/best.pt")
                self.person_model = YOLO("yolov8n.pt")

        self.valid_classes = ["Pot"]
        self.selected_class = "Pot"

        self.depth_image = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.target_frame = "link_base"

        # EMA filter used only for the official one-shot result
        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None
        self.alpha = 0.2

        # Latest official one-shot detection storage
        self.latest_camera_target = None
        self.latest_target_base = None
        self.latest_class = None
        self.latest_detection_time = 0.0

        # One-shot service state
        self.request_lock = threading.Lock()
        self.request_active = False
        self.request_class = None
        self.request_target_frame = None
        self.request_warmup_remaining = 0
        self.request_success = False
        self.request_message = ""
        self.request_detected_target = None
        self.request_target = None

        self.service_base_timeout_sec = 10.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Keep old topic names
        self.selected_class_pub = self.create_publisher(
            String,
            "/selected_class",
            10
        )

        self.detected_target_pub = self.create_publisher(
            PointStamped,
            "/detected_target",
            10
        )

        self.target_base_pub = self.create_publisher(
            PointStamped,
            "/target_base",
            10
        )

        self.annotated_image_pub = self.create_publisher(
            Image,
            "/visionv1/annotated_image",
            10
        )

        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.image_callback,
            sensor_qos,
            callback_group=self.callback_group
        )

        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            sensor_qos,
            callback_group=self.callback_group
        )

        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            sensor_qos,
            callback_group=self.callback_group
        )

        # Allows terminal / Flutter / other nodes to still change preview class by topic
        self.create_subscription(
            String,
            "/selected_class",
            self.selected_class_callback,
            10,
            callback_group=self.callback_group
        )

        self.detect_service = self.create_service(
            VisionDetect,
            "/visionv1/detect",
            self.detect_service_callback,
            callback_group=self.callback_group
        )

        # Keep publishing selected class so old topic behaviour remains
        self.create_timer(
            0.5,
            self.publish_selected_class,
            callback_group=self.callback_group
        )

        self.get_logger().info(f"Model classes: {self.model.names}")
        self.get_logger().info("VisionV1 node started")
        self.get_logger().info("Continuous preview: /visionv1/annotated_image")
        self.get_logger().info("One-shot service: /visionv1/detect")
        self.get_logger().info("/detected_target and /target_base publish only when service is called")

    def publish_selected_class(self):
        if self.shutting_down or not rclpy.ok():
            return

        msg = String()
        msg.data = self.selected_class

        try:
            self.selected_class_pub.publish(msg)
        except Exception:
            return

    def selected_class_callback(self, msg):
        class_name = msg.data.strip()

        if class_name not in self.valid_classes:
            self.get_logger().warn(f"Invalid class selected: {class_name}")
            return

        if class_name != self.selected_class:
            self.selected_class = class_name
            self.clear_latest_detection()
            self.get_logger().info(f"Selected class updated: {self.selected_class}")

    def clear_latest_detection(self):
        self.latest_camera_target = None
        self.latest_target_base = None
        self.latest_class = None
        self.latest_detection_time = 0.0
        self.filtered_x = None
        self.filtered_y = None
        self.filtered_z = None

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
            conf=0.9,
            classes=[0],
            verbose=False
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

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

    def consume_warmup_frame(self):
        with self.request_lock:
            if not self.request_active:
                return

            if self.request_warmup_remaining > 0:
                self.request_warmup_remaining -= 1

                if (
                    self.request_warmup_remaining == 0 or
                    self.request_warmup_remaining % 10 == 0
                ):
                    self.get_logger().info(
                        f"Warmup frames remaining: {self.request_warmup_remaining}"
                    )

    def service_request_ready(self):
        with self.request_lock:
            return self.request_active and self.request_warmup_remaining <= 0

    def finish_service_request_if_ready(self, detected_target):
        with self.request_lock:
            if not self.request_active:
                return

            if self.request_warmup_remaining > 0:
                return

            if self.request_class != self.selected_class:
                return

            requested_frame = self.request_target_frame

        target = self.transform_target(
            detected_target,
            requested_frame,
            use_filter=True
        )

        if target is None:
            # Keep waiting until TF becomes available or service times out
            return

        self.detected_target_pub.publish(detected_target)
        self.target_base_pub.publish(target)

        with self.request_lock:
            if not self.request_active:
                return

            self.latest_camera_target = detected_target
            self.latest_target_base = target
            self.latest_class = self.selected_class
            self.latest_detection_time = time.time()

            self.request_success = True
            self.request_message = "Detection successful"
            self.request_detected_target = detected_target
            self.request_target = target
            self.request_active = False

        self.get_logger().info(
            f"One-shot detection complete. Target in {requested_frame}: "
            f"X={target.point.x:.3f}, Y={target.point.y:.3f}, Z={target.point.z:.3f}"
        )

    def image_callback(self, msg):
        if self.shutting_down or not rclpy.ok():
            return

        missing = []

        if self.depth_image is None:
            missing.append("depth_image")

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            missing.append("camera_info")

        if missing:
            return

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            "bgr8"
        )

        annotated = frame.copy()

        # Check whether a service request is currently active.
        # If no service request is active, do NOT run YOLO.
        with self.request_lock:
            request_active = self.request_active
            request_class = self.request_class
            warmup_remaining = self.request_warmup_remaining

        if not request_active:
            cv2.putText(
                annotated,
                "Vision idle - waiting for service request",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8"
            )
            annotated_msg.header = msg.header

            try:
                self.annotated_image_pub.publish(annotated_msg)
            except Exception:
                return

            return

        # Service request is active, so count warmup frames first.
        self.consume_warmup_frame()

        with self.request_lock:
            warmup_remaining = self.request_warmup_remaining
            request_class = self.request_class

        if warmup_remaining > 0:
            cv2.putText(
                annotated,
                f"Service request active | Warmup remaining: {warmup_remaining}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding="bgr8"
            )
            annotated_msg.header = msg.header

            try:
                self.annotated_image_pub.publish(annotated_msg)
            except Exception:
                return

            return

        # Only after service request + warmup, run YOLO.
        results = self.model(
            frame,
            conf=0.4,
            verbose=False
        )

        found_target = False
        detected_target = None

        for box in results[0].boxes:
            detected_class_id = int(box.cls[0])
            detected_class_name = self.model.names[detected_class_id]
            confidence = float(box.conf[0].item())

            if detected_class_name != request_class:
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
                continue

            depth_crop = self.depth_image[
                v - 5:v + 6,
                u - 5:u + 6
            ]

            valid_depths = depth_crop[
                np.isfinite(depth_crop) & (depth_crop > 0)
            ]

            if valid_depths.size == 0:
                continue

            depth_raw = np.median(valid_depths)
            depth_m = self.depth_to_meters(depth_raw)

            if depth_m <= 0.0 or depth_m > 5.0:
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

            detected_target = PointStamped()
            detected_target.header.stamp = msg.header.stamp
            detected_target.header.frame_id = "camera_color_frame"
            detected_target.point.x = ros_x
            detected_target.point.y = ros_y
            detected_target.point.z = ros_z

            cv2.circle(
                annotated,
                (u, v),
                5,
                (0, 255, 0),
                -1
            )

            text_lines = [
                f"{request_class}",
                f"Conf: {confidence * 100:.1f}%",
                f"ROS Cam X: {ros_x:.3f} m",
                f"ROS Cam Y: {ros_y:.3f} m",
                f"ROS Cam Z: {ros_z:.3f} m"
            ]

            text_x = x1_i
            text_y = max(25, y1_i - 125)

            self.draw_xyz_text(
                annotated,
                text_lines,
                text_x,
                text_y
            )

            # Use the first valid matching target only
            break

        if not found_target:
            cv2.putText(
                annotated,
                f"Service request active | No {request_class} found",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Official robot result only happens during service request.
        if detected_target is not None:
            self.finish_service_request_if_ready(detected_target)

        annotated_msg = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding="bgr8"
        )
        annotated_msg.header = msg.header

        try:
            self.annotated_image_pub.publish(annotated_msg)
        except Exception:
            return

    def copy_point_stamped(self, msg, frame_id=None):
        copied = PointStamped()
        copied.header.stamp = self.get_clock().now().to_msg()
        copied.header.frame_id = frame_id if frame_id is not None else msg.header.frame_id
        copied.point.x = msg.point.x
        copied.point.y = msg.point.y
        copied.point.z = msg.point.z
        return copied

    def transform_target(self, detected_target, target_frame, use_filter=True):
        source_frame = detected_target.header.frame_id

        if source_frame == "":
            self.get_logger().warn("Rejected target: empty frame_id")
            return None

        if target_frame == "" or target_frame is None:
            target_frame = self.target_frame

        if source_frame == target_frame:
            return self.copy_point_stamped(detected_target, frame_id=target_frame)

        if not self.tf_buffer.can_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=0.0)
        ):
            self.get_logger().warn(
                f"TF not ready yet from {source_frame} to {target_frame}.",
                throttle_duration_sec=2.0
            )
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

            target = do_transform_point(detected_target, transform)
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = target_frame

            if use_filter:
                if self.filtered_x is None:
                    self.filtered_x = target.point.x
                    self.filtered_y = target.point.y
                    self.filtered_z = target.point.z
                else:
                    self.filtered_x = self.alpha * target.point.x + (1 - self.alpha) * self.filtered_x
                    self.filtered_y = self.alpha * target.point.y + (1 - self.alpha) * self.filtered_y
                    self.filtered_z = self.alpha * target.point.z + (1 - self.alpha) * self.filtered_z

                target.point.x = self.filtered_x
                target.point.y = self.filtered_y
                target.point.z = self.filtered_z

            return target

        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed from {source_frame} to {target_frame}: {e}",
                throttle_duration_sec=2.0
            )
            return None

    def detect_service_callback(self, request, response):
        requested_class = request.selected_class.strip()
        requested_frame = request.target_frame.strip()
        warmup_frames = int(request.warmup_frames)

        if requested_frame == "":
            requested_frame = self.target_frame

        if warmup_frames < 0:
            warmup_frames = 0

        if warmup_frames > 100:
            warmup_frames = 100

        if requested_class not in self.valid_classes:
            response.success = False
            response.message = (
                f"Invalid selected_class '{requested_class}'. "
                f"Use one of: {self.valid_classes}"
            )
            return response

        self.selected_class = requested_class
        self.publish_selected_class()
        self.clear_latest_detection()

        with self.request_lock:
            self.request_active = True
            self.request_class = requested_class
            self.request_target_frame = requested_frame
            self.request_warmup_remaining = warmup_frames
            self.request_success = False
            self.request_message = ""
            self.request_detected_target = None
            self.request_target = None

        timeout_sec = max(
            self.service_base_timeout_sec,
            warmup_frames * 0.8 + 8.0
        )

        self.get_logger().info(
            f"Service request received: class={requested_class}, "
            f"target_frame={requested_frame}, warmup_frames={warmup_frames}, "
            f"timeout={timeout_sec:.1f}s"
        )

        start_time = time.time()

        while rclpy.ok():
            with self.request_lock:
                done = self.request_success
                detected_target = self.request_detected_target
                target = self.request_target
                message = self.request_message

            if done:
                response.success = True
                response.message = message
                response.detected_target = detected_target
                response.target = target
                return response

            if time.time() - start_time > timeout_sec:
                with self.request_lock:
                    self.request_active = False

                response.success = False
                response.message = (
                    f"Timeout. No valid {requested_class} detected after "
                    f"{warmup_frames} warmup frames within {timeout_sec:.1f}s."
                )
                return response

            time.sleep(0.05)

        with self.request_lock:
            self.request_active = False

        response.success = False
        response.message = "ROS stopped before detection completed."
        return response


def main(args=None):
    rclpy.init(args=args)

    node = VisionV1Node()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutting_down = True
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
