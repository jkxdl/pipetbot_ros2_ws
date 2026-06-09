from pathlib import Path

from setuptools import setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="pipetbot-ros2-workspace",
    version="0.1.0",
    description="ROS 2 workspace for a pipetting robot with perception, control, simulation, GUI, and reinforcement learning.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="robot",
    license="Proprietary",
    py_modules=[],
    include_package_data=False,
    install_requires=[
        "numpy",
        "opencv-python",
        "PyYAML",
        "gymnasium",
        "skrl>=1.4.2",
        "torch",
        "ultralytics",
        "trimesh",
        "python-fcl",
        "urdf-parser-py",
        "plyfile",
        "scipy",
        "packaging",
        "PySide2",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
        "docs": [
            "markdown",
        ],
    },
    python_requires=">=3.10",
)
