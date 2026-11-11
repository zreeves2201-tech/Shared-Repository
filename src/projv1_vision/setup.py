from setuptools import find_packages, setup

package_name = 'projv1_vision'

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
		'yolov2_node = projv1_vision.yolov2_node:main',
		'target_transform_node = projv1_vision.target_transform_node:main',
		'select_class_node = projv1_vision.select_class_node:main',
                'visionv1_node = projv1_vision.visionv1_node:main',
                'visionv1_viewer = projv1_vision.visionv1_viewer:main',
        ],
    },
)
