from setuptools import find_packages, setup

package_name = 'circle_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Stereo Depth Circle Detector with Kalman Filtering',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'circle_pub = circle_detection.circle_pub:main',
            'circleAll_detector = circle_detection.circleAll_detector:main',
            'circle_test = circle_detection.circle_test:main'
        ],
    },
)

