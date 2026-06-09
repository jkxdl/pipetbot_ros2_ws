import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arm_action'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
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
        'action_execute = arm_action.action_execute:main',
        'pipetting_control = arm_action.pipetting_control:main',
        'jointaction_execute = arm_action.jointaction_execute:main',
        'quintic_spline = arm_action.quintic_spline:main',
        'planning_scene_initializer = arm_action.planning_scene_initializer:main'
        ],
    },
)
