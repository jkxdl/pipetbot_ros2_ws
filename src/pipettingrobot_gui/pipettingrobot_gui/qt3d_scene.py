from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from pipettingrobot_interfaces.msg import PipettingSceneState
from PySide2.Qt3DCore import Qt3DCore
from PySide2.Qt3DExtras import Qt3DExtras
from PySide2.Qt3DRender import Qt3DRender
from PySide2.QtCore import QObject, QTimer, QUrl
from PySide2.QtGui import QColor, QVector3D, QQuaternion
from tf2_ros import Buffer, TransformListener


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def package_uri_to_path(uri: str) -> str:
    if uri.startswith('package://'):
        body = uri[len('package://') :]
        package_name, relative_path = body.split('/', 1)
        return f"{get_package_share_directory(package_name)}/{relative_path}"
    return uri


def point_to_vector(point: Point) -> QVector3D:
    return QVector3D(point.x, point.y, point.z)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> QQuaternion:
    qx = QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), math.degrees(roll))
    qy = QQuaternion.fromAxisAndAngle(QVector3D(0.0, 1.0, 0.0), math.degrees(pitch))
    qz = QQuaternion.fromAxisAndAngle(QVector3D(0.0, 0.0, 1.0), math.degrees(yaw))
    return qz * qy * qx


def apply_transform(
    transform: Qt3DCore.QTransform,
    translation: QVector3D,
    scale: QVector3D,
    rotation: Optional[QQuaternion] = None,
):
    transform.setTranslation(translation)
    transform.setScale3D(scale)
    transform.setRotation(rotation or QQuaternion())


@dataclass
class RobotLinkVisual:
    name: str
    mesh_path: str
    entity: Qt3DCore.QEntity
    transform: Qt3DCore.QTransform
    fallback_entity: Qt3DCore.QEntity
    fallback_transform: Qt3DCore.QTransform


@dataclass
class EndEffectorVisualSpec:
    mesh_uri: str
    offset_xyz: QVector3D
    offset_rotation: QQuaternion


class PipettingScene3D(QObject):
    ROBOT_LINK_NAMES = [
        'aubo_base',
        'shoulder_Link',
        'upperArm_Link',
        'foreArm_Link',
        'wrist1_Link',
        'wrist2_Link',
        'wrist3_Link',
    ]
    DEFAULT_ROBOT_LAYOUT = {
        'aubo_base': (QVector3D(0.0, 0.0, 0.02), QVector3D(1.0, 1.0, 1.0)),
        'shoulder_Link': (QVector3D(0.0, 0.0, 0.18), QVector3D(1.0, 1.0, 1.0)),
        'upperArm_Link': (QVector3D(0.02, 0.0, 0.38), QVector3D(1.0, 1.0, 1.0)),
        'foreArm_Link': (QVector3D(0.10, 0.0, 0.56), QVector3D(1.0, 1.0, 1.0)),
        'wrist1_Link': (QVector3D(0.16, 0.0, 0.70), QVector3D(1.0, 1.0, 1.0)),
        'wrist2_Link': (QVector3D(0.21, 0.0, 0.79), QVector3D(1.0, 1.0, 1.0)),
        'wrist3_Link': (QVector3D(0.25, 0.0, 0.87), QVector3D(1.0, 1.0, 1.0)),
    }

    def __init__(self, ros_node, parent=None):
        super().__init__(parent)
        self.ros_node = ros_node
        self.window = Qt3DExtras.Qt3DWindow()
        self.window.defaultFrameGraph().setClearColor(QColor(34, 36, 40))
        self.root = Qt3DCore.QEntity()
        self.window.setRootEntity(self.root)
        self._logged_mesh_paths = set()
        self._logged_tf_frames = set()
        self._tf_seen = set()
        self._ee_logged = False
        self.end_effector_spec: Optional[EndEffectorVisualSpec] = None

        self.camera = self.window.camera()
        self.camera.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.01, 100.0)
        self.camera.setPosition(QVector3D(-1.25, 1.35, 0.95))
        self.camera.setViewCenter(QVector3D(0.0, 0.0, 0.18))

        self.camera_controller = Qt3DExtras.QOrbitCameraController(self.root)
        self.camera_controller.setCamera(self.camera)
        self.camera_controller.setLinearSpeed(50.0)
        self.camera_controller.setLookSpeed(180.0)

        self.declare_defaults()
        self.end_effector_spec = self.load_end_effector_visual_spec()
        self.setup_lighting()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.ros_node)

        self.ground_entity = self.make_ground_entity()
        self.axis_entity = self.make_axis_entity()
        self.rack_entity, self.rack_transform = self.make_mesh_or_box_entity(
            mesh_uri=self.ros_node.get_parameter('rack_mesh').value,
            fallback_size=QVector3D(0.18, 0.08, 0.10),
            color=QColor(190, 190, 208, 180),
        )
        self.beaker_entity, self.beaker_transform = self.make_mesh_or_cylinder_entity(
            mesh_uri=self.ros_node.get_parameter('beaker_mesh').value,
            fallback_radius=0.035,
            fallback_height=0.09,
            color=QColor(230, 165, 56, 140),
        )

        self.tube_entities: Dict[int, Dict[str, object]] = {}
        self.robot_links = self.build_robot_visuals()
        self.end_effector_entity, self.end_effector_transform = self.build_end_effector_visual()

        self.scene_timer = QTimer(self)
        self.scene_timer.timeout.connect(self.sync_scene_state)
        self.scene_timer.start(100)

        self.tf_timer = QTimer(self)
        self.tf_timer.timeout.connect(self.sync_robot_tf)
        self.tf_timer.start(80)

    def declare_defaults(self):
        defaults = {
            'rack_mesh': 'package://pipettingrobot_gui/meshes/TestTubeRack.dae',
            'beaker_mesh': 'package://pipettingrobot_gui/meshes/Beaker.dae',
            'tube_mesh': 'package://pipettingrobot_gui/meshes/TestTube.dae',
            'aubo_type': 'aubo_C5',
            'robot_base_frame': 'base_link',
            'end_effector_frame': 'tool0',
            'end_effector_urdf': 'package://pipettingrobot_sim/urdf/C5.urdf.xacro',
            'end_effector_link': 'tool0',
        }
        for key, value in defaults.items():
            if not self.ros_node.has_parameter(key):
                self.ros_node.declare_parameter(key, value)

    def create_container(self, parent):
        from PySide2.QtWidgets import QWidget

        return QWidget.createWindowContainer(self.window, parent)

    def setup_lighting(self):
        self.light_entity = Qt3DCore.QEntity(self.root)
        self.light = Qt3DRender.QPointLight(self.light_entity)
        self.light.setColor(QColor(255, 255, 245))
        self.light.setIntensity(1.6)
        self.light_transform = Qt3DCore.QTransform()
        self.light_transform.setTranslation(QVector3D(-1.5, 1.8, 2.6))
        self.light_entity.addComponent(self.light)
        self.light_entity.addComponent(self.light_transform)

        self.fill_light_entity = Qt3DCore.QEntity(self.root)
        self.fill_light = Qt3DRender.QPointLight(self.fill_light_entity)
        self.fill_light.setColor(QColor(180, 195, 255))
        self.fill_light.setIntensity(0.8)
        self.fill_light_transform = Qt3DCore.QTransform()
        self.fill_light_transform.setTranslation(QVector3D(1.2, 1.2, 1.6))
        self.fill_light_entity.addComponent(self.fill_light)
        self.fill_light_entity.addComponent(self.fill_light_transform)

    def resolve_mesh_path(self, mesh_uri: str) -> str:
        if not mesh_uri:
            return ''
        mesh_path = package_uri_to_path(mesh_uri)
        if mesh_path not in self._logged_mesh_paths:
            if os.path.exists(mesh_path):
                self.ros_node.get_logger().info(f'Loaded mesh path: {mesh_path}')
            else:
                self.ros_node.get_logger().warn(f'Mesh path not found: {mesh_path}')
            self._logged_mesh_paths.add(mesh_path)
        return mesh_path

    def load_end_effector_visual_spec(self) -> Optional[EndEffectorVisualSpec]:
        urdf_uri = self.ros_node.get_parameter('end_effector_urdf').value
        link_name = self.ros_node.get_parameter('end_effector_link').value
        urdf_path = package_uri_to_path(urdf_uri)
        if not os.path.exists(urdf_path):
            self.ros_node.get_logger().warn(f'End effector URDF path not found: {urdf_path}')
            return None

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for link in root.findall('link'):
                if link.get('name') != link_name:
                    continue
                visual = link.find('visual')
                if visual is None:
                    continue
                geometry = visual.find('geometry')
                mesh = geometry.find('mesh') if geometry is not None else None
                if mesh is None:
                    continue
                mesh_uri = mesh.get('filename', '')
                origin = visual.find('origin')
                xyz = QVector3D(0.0, 0.0, 0.0)
                rotation = QQuaternion()
                if origin is not None:
                    xyz_text = origin.get('xyz', '0 0 0').split()
                    if len(xyz_text) == 3:
                        xyz = QVector3D(float(xyz_text[0]), float(xyz_text[1]), float(xyz_text[2]))
                    rpy_text = origin.get('rpy', '0 0 0')
                    if '${' not in rpy_text:
                        rpy = [float(v) for v in rpy_text.split()]
                        if len(rpy) == 3:
                            rotation = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
                    else:
                        # Match the current tool0 visual origin in C5.urdf.xacro
                        rotation = rpy_to_quaternion(-math.pi / 2.0, math.pi, 0.0)
                self.ros_node.get_logger().info(
                    f'Loaded end effector visual from URDF: link={link_name}, mesh={mesh_uri}'
                )
                return EndEffectorVisualSpec(
                    mesh_uri=mesh_uri,
                    offset_xyz=xyz,
                    offset_rotation=rotation,
                )
        except Exception as exc:
            self.ros_node.get_logger().warn(f'Failed to parse end effector URDF: {exc}')
        return None

    def make_ground_entity(self):
        entity = Qt3DCore.QEntity(self.root)
        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(2.0)
        mesh.setYExtent(2.0)
        mesh.setZExtent(0.002)
        material = Qt3DExtras.QPhongMaterial(entity)
        material.setDiffuse(QColor(75, 78, 84))
        transform = Qt3DCore.QTransform()
        transform.setTranslation(QVector3D(0.0, 0.0, -0.001))
        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        return entity

    def make_axis_entity(self):
        axis_root = Qt3DCore.QEntity(self.root)
        self.make_axis_line(axis_root, QVector3D(0.25, 0.004, 0.004), QVector3D(0.125, 0.0, 0.0), QColor(210, 70, 70))
        self.make_axis_line(axis_root, QVector3D(0.004, 0.25, 0.004), QVector3D(0.0, 0.125, 0.0), QColor(70, 190, 90))
        self.make_axis_line(axis_root, QVector3D(0.004, 0.004, 0.25), QVector3D(0.0, 0.0, 0.125), QColor(70, 120, 220))
        return axis_root

    def make_axis_line(self, parent, size, translation, color):
        entity = Qt3DCore.QEntity(parent)
        mesh = Qt3DExtras.QCuboidMesh()
        mesh.setXExtent(size.x())
        mesh.setYExtent(size.y())
        mesh.setZExtent(size.z())
        material = Qt3DExtras.QPhongMaterial(entity)
        material.setDiffuse(color)
        transform = Qt3DCore.QTransform()
        transform.setTranslation(translation)
        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)

    def make_robot_fallback_entity(self, link_name: str):
        entity = Qt3DCore.QEntity(self.root)
        mesh = Qt3DExtras.QCuboidMesh()
        if link_name == 'aubo_base':
            mesh.setXExtent(0.18)
            mesh.setYExtent(0.18)
            mesh.setZExtent(0.06)
            color = QColor(90, 110, 140)
        else:
            mesh.setXExtent(0.05)
            mesh.setYExtent(0.05)
            mesh.setZExtent(0.14)
            color = QColor(255, 160, 80)
        material = Qt3DExtras.QPhongMaterial(entity)
        material.setDiffuse(color)
        transform = Qt3DCore.QTransform()
        entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        return entity, transform

    def make_mesh_or_box_entity(self, mesh_uri, fallback_size, color):
        entity = Qt3DCore.QEntity(self.root)
        transform = Qt3DCore.QTransform()
        material = Qt3DExtras.QPhongAlphaMaterial(entity)
        material.setDiffuse(color)
        if mesh_uri:
            mesh_path = self.resolve_mesh_path(mesh_uri)
            mesh = Qt3DRender.QSceneLoader(entity)
            mesh.setSource(QUrl.fromLocalFile(mesh_path))
            transform.setScale3D(QVector3D(1.0, 1.0, 1.0))
            entity.addComponent(mesh)
        else:
            mesh = Qt3DExtras.QCuboidMesh()
            mesh.setXExtent(fallback_size.x())
            mesh.setYExtent(fallback_size.y())
            mesh.setZExtent(fallback_size.z())
            entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        return entity, transform

    def make_mesh_or_cylinder_entity(self, mesh_uri, fallback_radius, fallback_height, color):
        entity = Qt3DCore.QEntity(self.root)
        transform = Qt3DCore.QTransform()
        material = Qt3DExtras.QPhongAlphaMaterial(entity)
        material.setDiffuse(color)
        if mesh_uri:
            mesh_path = self.resolve_mesh_path(mesh_uri)
            mesh = Qt3DRender.QSceneLoader(entity)
            mesh.setSource(QUrl.fromLocalFile(mesh_path))
            transform.setScale3D(QVector3D(1.0, 1.0, 1.0))
            entity.addComponent(mesh)
        else:
            mesh = Qt3DExtras.QCylinderMesh()
            mesh.setRadius(fallback_radius)
            mesh.setLength(fallback_height)
            entity.addComponent(mesh)
        entity.addComponent(material)
        entity.addComponent(transform)
        return entity, transform

    def create_tube_visual(self, index: int):
        tube_mesh_uri = self.ros_node.get_parameter('tube_mesh').value
        tube_radius = 0.006
        tube_height = 0.075

        wall = Qt3DCore.QEntity(self.root)
        wall_material = Qt3DExtras.QPhongAlphaMaterial(wall)
        wall_material.setDiffuse(QColor(190, 210, 240, 110))
        wall_transform = Qt3DCore.QTransform()
        if tube_mesh_uri:
            wall_mesh = Qt3DRender.QSceneLoader(wall)
            wall_mesh.setSource(QUrl.fromLocalFile(self.resolve_mesh_path(tube_mesh_uri)))
            wall_transform.setScale3D(QVector3D(1.0, 1.0, 1.0))
            wall.addComponent(wall_mesh)
        else:
            wall_mesh = Qt3DExtras.QCylinderMesh()
            wall_mesh.setRadius(tube_radius)
            wall_mesh.setLength(tube_height)
            wall.addComponent(wall_mesh)
        wall.addComponent(wall_material)
        wall.addComponent(wall_transform)

        liquid = Qt3DCore.QEntity(self.root)
        liquid_mesh = Qt3DExtras.QCylinderMesh()
        liquid_mesh.setRadius(tube_radius * 0.72)
        liquid_mesh.setLength(0.001)
        liquid_material = Qt3DExtras.QPhongAlphaMaterial(liquid)
        liquid_material.setDiffuse(QColor(60, 150, 245, 180))
        liquid_transform = Qt3DCore.QTransform()
        liquid.addComponent(liquid_mesh)
        liquid.addComponent(liquid_material)
        liquid.addComponent(liquid_transform)

        tip = Qt3DCore.QEntity(self.root)
        tip_mesh = Qt3DExtras.QConeMesh()
        tip_mesh.setTopRadius(0.0)
        tip_mesh.setBottomRadius(tube_radius * 1.25)
        tip_mesh.setLength(0.02)
        tip_material = Qt3DExtras.QPhongMaterial(tip)
        tip_material.setDiffuse(QColor(255, 216, 60))
        tip_transform = Qt3DCore.QTransform()
        tip.setEnabled(False)
        tip.addComponent(tip_mesh)
        tip.addComponent(tip_material)
        tip.addComponent(tip_transform)

        self.tube_entities[index] = {
            'wall': wall,
            'wall_transform': wall_transform,
            'wall_material': wall_material,
            'liquid': liquid,
            'liquid_mesh': liquid_mesh,
            'liquid_transform': liquid_transform,
            'tip': tip,
            'tip_transform': tip_transform,
        }

    def build_robot_visuals(self):
        visuals: Dict[str, RobotLinkVisual] = {}
        aubo_type = self.ros_node.get_parameter('aubo_type').value
        base_dir = get_package_share_directory('aubo_description')
        for index, link_name in enumerate(self.ROBOT_LINK_NAMES):
            mesh_path = f'{base_dir}/meshes/{aubo_type}/visual/link{index}.DAE'
            entity = Qt3DCore.QEntity(self.root)
            self.resolve_mesh_path(mesh_path)
            scene_loader = Qt3DRender.QSceneLoader(entity)
            scene_loader.setSource(QUrl.fromLocalFile(mesh_path))
            transform = Qt3DCore.QTransform()
            entity.addComponent(scene_loader)
            entity.addComponent(transform)
            fallback_entity, fallback_transform = self.make_robot_fallback_entity(link_name)
            default_translation, default_scale = self.DEFAULT_ROBOT_LAYOUT.get(
                link_name,
                (QVector3D(0.0, 0.0, 0.1 * index), QVector3D(1.0, 1.0, 1.0)),
            )
            apply_transform(
                transform,
                default_translation,
                default_scale,
            )
            apply_transform(
                fallback_transform,
                default_translation,
                QVector3D(1.0, 1.0, 1.0),
            )
            visuals[link_name] = RobotLinkVisual(
                name=link_name,
                mesh_path=mesh_path,
                entity=entity,
                transform=transform,
                fallback_entity=fallback_entity,
                fallback_transform=fallback_transform,
            )
        return visuals

    def build_end_effector_visual(self):
        mesh_uri = self.end_effector_spec.mesh_uri if self.end_effector_spec else ''
        entity = Qt3DCore.QEntity(self.root)
        transform = Qt3DCore.QTransform()
        if mesh_uri:
            mesh_path = self.resolve_mesh_path(mesh_uri)
            mesh = Qt3DRender.QSceneLoader(entity)
            mesh.setSource(QUrl.fromLocalFile(mesh_path))
            entity.addComponent(mesh)
        else:
            mesh = Qt3DExtras.QCuboidMesh()
            mesh.setXExtent(0.04)
            mesh.setYExtent(0.20)
            mesh.setZExtent(0.04)
            entity.addComponent(mesh)
        material = Qt3DExtras.QPhongMaterial(entity)
        material.setDiffuse(QColor(232, 232, 240))
        entity.addComponent(material)
        entity.addComponent(transform)
        entity.setEnabled(False)
        return entity, transform

    def sync_scene_state(self):
        state: PipettingSceneState = self.ros_node.scene_state

        self.rack_entity.setEnabled(state.rack_detected)
        if state.rack_detected:
            apply_transform(
                self.rack_transform,
                point_to_vector(state.rack_origin) + QVector3D(0.0, 0.0, -0.05),
                self.rack_transform.scale3D() if self.rack_transform.scale3D().length() > 0.0 else QVector3D(1.0, 1.0, 1.0),
            )

        self.beaker_entity.setEnabled(state.beaker_detected)
        if state.beaker_detected:
            apply_transform(
                self.beaker_transform,
                point_to_vector(state.beaker_origin) + QVector3D(0.0, 0.0, -0.045),
                self.beaker_transform.scale3D() if self.beaker_transform.scale3D().length() > 0.0 else QVector3D(1.0, 1.0, 1.0),
            )

        seen = set()
        for tube in state.tubes:
            seen.add(tube.index)
            if tube.index not in self.tube_entities:
                self.create_tube_visual(tube.index)
            self.update_tube_visual(
                tube.index,
                tube.opening,
                tube.filled,
                tube.fill_ratio,
                tube.index == state.active_tube_index,
            )

        for index, visual in self.tube_entities.items():
            enabled = index in seen
            visual['wall'].setEnabled(enabled)
            visual['liquid'].setEnabled(enabled)
            visual['tip'].setEnabled(enabled and index == state.active_tube_index)

    def update_tube_visual(self, index, opening, filled, fill_ratio, active):
        visual = self.tube_entities[index]
        tube_height = 0.075
        tube_radius = 0.006
        opening_vec = point_to_vector(opening)

        wall_color = QColor(220, 230, 250, 150 if active else 95)
        visual['wall_material'].setDiffuse(wall_color)
        apply_transform(
            visual['wall_transform'],
            opening_vec + QVector3D(0.0, 0.0, -tube_height / 2.0),
            visual['wall_transform'].scale3D(),
        )

        fill = clamp(fill_ratio if filled else 0.0, 0.0, 1.0)
        liquid_height = max(0.001, tube_height * 0.60 * fill)
        visual['liquid_mesh'].setLength(liquid_height)
        liquid_z = opening.z - tube_height + (tube_height * 0.05) + liquid_height / 2.0
        apply_transform(
            visual['liquid_transform'],
            QVector3D(opening.x, opening.y, liquid_z),
            QVector3D(1.0, 1.0, 1.0),
        )
        visual['liquid'].setEnabled(fill > 0.001)

        apply_transform(
            visual['tip_transform'],
            opening_vec + QVector3D(0.0, 0.0, 0.02),
            QVector3D(1.0, 1.0, 1.0),
            QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), 90.0),
        )
        visual['tip'].setEnabled(active)

    def sync_robot_tf(self):
        base_frame = self.ros_node.get_parameter('robot_base_frame').value
        for link_name, visual in self.robot_links.items():
            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame,
                    link_name,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01),
                )
            except Exception:
                if link_name not in self._logged_tf_frames:
                    self.ros_node.get_logger().warn(
                        f'No TF yet for {base_frame} -> {link_name}; using default robot pose for now.'
                    )
                    self._logged_tf_frames.add(link_name)
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            q = QQuaternion(rotation.w, rotation.x, rotation.y, rotation.z)
            if link_name not in self._tf_seen:
                self.ros_node.get_logger().info(
                    f'TF received for {base_frame} -> {link_name}; switching link to live pose.'
                )
                self._tf_seen.add(link_name)
            apply_transform(
                visual.transform,
                QVector3D(translation.x, translation.y, translation.z),
                QVector3D(1.0, 1.0, 1.0),
                q,
            )
            apply_transform(
                visual.fallback_transform,
                QVector3D(translation.x, translation.y, translation.z),
                QVector3D(1.0, 1.0, 1.0),
                q,
            )

        ee_frame = self.ros_node.get_parameter('end_effector_frame').value
        try:
            ee_transform = self.tf_buffer.lookup_transform(
                base_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01),
            )
            ee_translation = ee_transform.transform.translation
            ee_rotation = ee_transform.transform.rotation
            ee_q = QQuaternion(
                ee_rotation.w,
                ee_rotation.x,
                ee_rotation.y,
                ee_rotation.z,
            )
            spec = self.end_effector_spec
            offset_xyz = spec.offset_xyz if spec is not None else QVector3D(0.0, 0.0, 0.0)
            offset_q = spec.offset_rotation if spec is not None else QQuaternion()
            visual_offset_t = ee_q.rotatedVector(offset_xyz)
            apply_transform(
                self.end_effector_transform,
                QVector3D(ee_translation.x, ee_translation.y, ee_translation.z)
                + visual_offset_t,
                QVector3D(1.0, 1.0, 1.0),
                ee_q * offset_q,
            )
            self.end_effector_entity.setEnabled(True)
            if not self._ee_logged:
                self.ros_node.get_logger().info(
                    f'End effector visual attached to {base_frame} -> {ee_frame}.'
                )
                self._ee_logged = True
        except Exception:
            self.end_effector_entity.setEnabled(False)

    def shutdown(self):
        self.scene_timer.stop()
        self.tf_timer.stop()
