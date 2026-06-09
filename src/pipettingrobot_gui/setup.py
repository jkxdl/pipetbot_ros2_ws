import os
from glob import glob

from setuptools import setup


package_name = 'pipettingrobot_gui'


setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*.dae')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
        (os.path.join('share', package_name, 'applications'), glob('applications/*.desktop')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='Operator panel and scene visualization for the pipetting robot.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'scene_bridge = pipettingrobot_gui.scene_bridge:main',
            'operator_panel = pipettingrobot_gui.operator_panel:main',
            'dae_viewer_test = pipettingrobot_gui.dae_viewer_test:main',
        ],
    },
)
