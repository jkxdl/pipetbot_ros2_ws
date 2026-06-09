from setuptools import find_packages, setup

package_name = 'rl_train'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','pytest'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'pointcloud = rl_train.pointcloud:main',
        'target_pose = rl_train.target_pose:main',
        'obstacle_feature_extractor = rl_train.obstacle_feature_extractor:main',
        'contact_sensor = rl_train.contact_sensor:main',
        'ee_pose_publisher = rl_train.ee_pose_publisher:main'
        ],
    },
)
