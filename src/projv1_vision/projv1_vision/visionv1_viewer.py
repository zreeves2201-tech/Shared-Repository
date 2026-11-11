import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class VisionV1Viewer(Node):

    def __init__(self):
        super().__init__('visionv1_viewer')

        self.bridge = CvBridge()
        self.latest_image = None
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            Image,
            '/visionv1/annotated_image',
            self.image_callback,
            sensor_qos
        )

        self.get_logger().info("VisionV1 viewer started")
        self.get_logger().info("Subscribing to /visionv1/annotated_image")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        with self.lock:
            self.latest_image = frame.copy()

    def show_image(self):
        with self.lock:
            if self.latest_image is None:
                return

            frame = self.latest_image.copy()

        cv2.namedWindow("VisionV1 YOLO Camera", cv2.WINDOW_NORMAL)
        cv2.imshow("VisionV1 YOLO Camera", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = VisionV1Viewer()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    spin_thread = threading.Thread(
        target=executor.spin,
        daemon=True
    )
    spin_thread.start()

    try:
        while rclpy.ok():
            node.show_image()
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
