from setuptools import setup
import os
from glob import glob

package_name = 'pipettingrobot_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 安装配置文件
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools','pytest'],
    zip_safe=True,
    maintainer='jkx',
    maintainer_email='2147765325@qq.com',
    description='Launch multiple nodes and launch files',
    license='Apache License 2.0',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)

