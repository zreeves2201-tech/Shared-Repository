import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading


class SelectClassNode(Node):

    def __init__(self):
        super().__init__('select_class_node')

        self.publisher_ = self.create_publisher(
            String,
            '/selected_class',
            10
        )

        self.selected_class = "Browncup"

        self.timer = self.create_timer(0.5, self.publish_class)

        self.get_logger().info("Select class node started")
        self.get_logger().info("Default selected: Browncup")

        self.input_thread = threading.Thread(
            target=self.ask_user,
            daemon=True
        )
        self.input_thread.start()

    def ask_user(self):

        while rclpy.ok():

            print("\nSelect object class:")
            print("1 = Browncup")
            print("2 = Whitecup")

            choice = input("Enter choice: ").strip()

            if choice == "1":
                self.selected_class = "Browncup"
                self.get_logger().info("Selected Browncup")

            elif choice == "2":
                self.selected_class = "Whitecup"
                self.get_logger().info("Selected Whitecup")

            else:
                self.get_logger().warn("Invalid choice. Enter 1 or 2.")

    def publish_class(self):

        msg = String()
        msg.data = self.selected_class
        self.publisher_.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = SelectClassNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
