from setuptools import find_packages, setup

package_name = 'target_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'yolo11_custom': ['cfg/models/11/*.yaml'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
    ],
    install_requires=['setuptools','pytest'],
    zip_safe=True,
    maintainer='jkx',
    maintainer_email='jkx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'yolov11_d435i = yolov11.yolov11_d435i:main',
        'yolov11_stereo = yolov11.yolov11_stereo:main',
        'stereo_imagepub = yolov11.stereo_imagepub:main',
        'd435i_pointcloud = yolov11.d435i_pointcloud:main'
        ],
    },
)
