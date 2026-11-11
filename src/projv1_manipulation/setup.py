from setuptools import find_packages, setup

package_name = 'projv1_manipulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
       
        'target_transform_node = projv1_manipulation.target_transform_node:main',
        'moveit_pick_place_node = projv1_manipulation.moveit_pick_place_node:main',
        'fake_moveit_pick_place_node = projv1_manipulation.fake_moveit_pick_place_node:main',
        'gazebo_cup_marker_node = projv1_manipulation.gazebo_cup_marker_node:main',
    ],
},
)
