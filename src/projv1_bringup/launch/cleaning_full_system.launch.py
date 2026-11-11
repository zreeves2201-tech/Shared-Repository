import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    xarm_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xarm_moveit_config'),
                'launch',
                '_robot_moveit_realmove.launch.py',
            )
        ),
        launch_arguments={
            'robot_type': 'xarm',
            'dof': '7',
            'robot_ip': '192.168.1.234',

            # Keep the existing gripper model. The physical brush extension is
            # represented explicitly by the brush_tip TF below.
            'add_other_geometry': 'true',
            'geometry_type': 'cylinder',

            # Brush dimensions.
            'geometry_height': '0.33',
            'geometry_radius': '0.045',
            'geometry_mass': '0.30',

            # Example: brush extends along the tool local Z-axis.
            # Cylinder origin is at its centre.
            'geometry_mesh_origin_xyz': '0 0 0.165',
            'geometry_mesh_origin_rpy': '0 0 0',

            # TCP at the physical tip of the brush.
            'geometry_mesh_tcp_xyz': '0 0 0.33',
            'geometry_mesh_tcp_rpy': '0 0 0',
        }.items(),
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
        }.items(),
    )

    camera_static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        output='screen',
        arguments=[
            '--x', '0.060',
            '--y', '0.000',
            '--z', '0.068',
            '--roll', '3.1416',
            '--pitch', '-1.5708',
            '--yaw', '0.0',
            '--frame-id', 'link_eef',
            '--child-frame-id', 'camera_link',
        ],
    )
    
    brush_tip_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='brush_tip_static_tf',
        output='screen',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.33',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            # The 33 cm actuator starts AFTER the existing gripper TCP.
            '--frame-id', 'link_tcp',
            '--child-frame-id', 'brush_tip',
        ],
    )

    vision_node = Node(
        package='projv1_vision',
        executable='visionv1_node',
        name='visionv1_node',
        output='screen',
    )

    vision_viewer = Node(
        package='projv1_vision',
        executable='visionv1_viewer',
        name='visionv1_viewer',
        output='screen',
    )

    cleaning_node = Node(
        package='projv1_manipulation',
        executable='dirty_spot_cleaning_node',
        name='dirty_spot_cleaning_node',
        output='screen',
    )

    return LaunchDescription([
        xarm_moveit,
        TimerAction(period=5.0, actions=[realsense]),
        TimerAction(period=8.0, actions=[camera_static_tf]),
        TimerAction(period=8.0, actions=[brush_tip_tf]),
        TimerAction(period=10.0, actions=[vision_node]),
        TimerAction(period=12.0, actions=[vision_viewer]),
        TimerAction(period=15.0, actions=[cleaning_node]),
    ])
