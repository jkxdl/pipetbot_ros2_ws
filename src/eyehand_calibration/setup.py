from setuptools import find_packages, setup

package_name = 'eyehand_calibration'

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
    #:tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'calibration_client = eyehand_calibration.calibration_client:main',
        'eyeinhand_calibration = eyehand_calibration.eyeinhand_calibration:main',
        'eyetohand_calibration = eyehand_calibration.eyetohand_calibration:main'
        ],
    },
)
