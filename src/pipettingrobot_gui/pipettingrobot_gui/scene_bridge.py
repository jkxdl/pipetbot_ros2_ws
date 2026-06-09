import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import Point
from pipettingrobot_interfaces.msg import PipettingSceneState, TubeVisualState
from pipettingrobot_interfaces.srv import SetActiveTube, SetTubeVisualState
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class TubeState:
    index: int
    opening: Point
    filled: bool = False
    fill_ratio: float = 0.0


class SceneBridge(Node):
    def __init__(self):
        super().__init__('pipetting_scene_bridge')

        self.declare_parameter('marker_input_topic', '/target_coordinates')
        self.declare_parameter('scene_marker_topic', '/pipetting_ui/scene_markers')
        self.declare_parameter('scene_state_topic', '/pipetting_ui/scene_state')
        self.declare_parameter('rack_mesh', '')
        self.declare_parameter('beaker_mesh', '')
        self.declare_parameter('tube_mesh', '')
        self.declare_parameter('rack_size', [0.18, 0.08, 0.10])
        self.declare_parameter('beaker_size', [0.07, 0.07, 0.09])
        self.declare_parameter('tube_radius', 0.006)
        self.declare_parameter('tube_height', 0.075)
        self.declare_parameter('liquid_color', [0.15, 0.55, 0.95, 0.85])

        marker_input_topic = self.get_parameter('marker_input_topic').value
        scene_marker_topic = self.get_parameter('scene_marker_topic').value
        scene_state_topic = self.get_parameter('scene_state_topic').value

        self.rack_point: Optional[Point] = None
        self.beaker_point: Optional[Point] = None
        self.tubes: Dict[int, TubeState] = {}
        self.active_tube_index = -1
        self.phase = 'idle'

        self.marker_sub = self.create_subscription(
            Marker, marker_input_topic, self.marker_callback, 20
        )
        self.scene_marker_pub = self.create_publisher(MarkerArray, scene_marker_topic, 10)
        self.scene_state_pub = self.create_publisher(PipettingSceneState, scene_state_topic, 10)

        self.create_service(Trigger, '/pipetting_ui/reset_scene', self.handle_reset_scene)
        self.create_service(SetActiveTube, '/pipetting_ui/set_active_tube', self.handle_set_active_tube)
        self.create_service(
            SetTubeVisualState,
            '/pipetting_ui/set_tube_visual_state',
            self.handle_set_tube_visual_state,
        )

        self.publish_timer = self.create_timer(0.5, self.publish_scene)
        self.get_logger().info('Scene bridge ready. Listening for visualization markers.')

    def marker_callback(self, marker: Marker) -> None:
        if marker.type != Marker.TEXT_VIEW_FACING or not marker.text:
            return

        label = marker.text.strip()
        point = Point(
            x=marker.pose.position.x,
            y=marker.pose.position.y,
            z=marker.pose.position.z - 0.01,
        )

        if label == 'TestTubeRack':
            self.rack_point = point
            self.phase = 'rack_detected'
            return

        if label == 'Beaker':
            self.beaker_point = point
            self.phase = 'beaker_detected'
            return

        match = re.fullmatch(r'T_(\d+)', label)
        if not match:
            return

        index = int(match.group(1)) - 1
        existing = self.tubes.get(index)
        filled = existing.filled if existing else False
        fill_ratio = existing.fill_ratio if existing else 0.0
        self.tubes[index] = TubeState(index=index, opening=point, filled=filled, fill_ratio=fill_ratio)
        self.phase = 'tubes_detected'

    def handle_reset_scene(self, request, response):
        del request
        self.rack_point = None
        self.beaker_point = None
        self.tubes.clear()
        self.active_tube_index = -1
        self.phase = 'idle'
        self.publish_scene(clear_only=True)
        response.success = True
        response.message = 'Scene reset.'
        return response

    def handle_set_active_tube(self, request, response):
        if request.tube_index < -1:
            response.success = False
            response.message = 'tube_index must be >= -1'
            return response

        self.active_tube_index = request.tube_index
        self.phase = 'highlighting'
        response.success = True
        response.message = f'Active tube set to {request.tube_index}'
        self.publish_scene()
        return response

    def handle_set_tube_visual_state(self, request, response):
        tube = self.tubes.get(request.tube_index)
        if tube is None:
            response.success = False
            response.message = f'Tube {request.tube_index} has not been detected yet.'
            return response

        tube.filled = request.filled
        tube.fill_ratio = max(0.0, min(1.0, request.fill_ratio))
        self.phase = 'tube_filled' if tube.filled else 'tube_cleared'
        response.success = True
        response.message = f'Tube {request.tube_index} visual state updated.'
        self.publish_scene()
        return response

    def publish_scene(self, clear_only: bool = False) -> None:
        marker_array = MarkerArray()
        if clear_only:
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            self.scene_marker_pub.publish(marker_array)
            self.scene_state_pub.publish(self.build_scene_state())
            return

        marker_array.markers.append(self.make_delete_all_marker())

        next_id = 0
        if self.rack_point is not None:
            rack_markers = self.make_rack_markers(self.rack_point, next_id)
            marker_array.markers.extend(rack_markers)
            next_id += len(rack_markers)

        if self.beaker_point is not None:
            beaker_markers = self.make_beaker_markers(self.beaker_point, next_id)
            marker_array.markers.extend(beaker_markers)
            next_id += len(beaker_markers)

        for index in sorted(self.tubes):
            tube_markers = self.make_tube_markers(self.tubes[index], next_id)
            marker_array.markers.extend(tube_markers)
            next_id += len(tube_markers)

        self.scene_marker_pub.publish(marker_array)
        self.scene_state_pub.publish(self.build_scene_state())

    def build_scene_state(self) -> PipettingSceneState:
        msg = PipettingSceneState()
        msg.rack_detected = self.rack_point is not None
        if self.rack_point is not None:
            msg.rack_origin = self.copy_point(self.rack_point)
        msg.beaker_detected = self.beaker_point is not None
        if self.beaker_point is not None:
            msg.beaker_origin = self.copy_point(self.beaker_point)
        msg.active_tube_index = self.active_tube_index
        msg.phase = self.phase

        for index in sorted(self.tubes):
            tube = self.tubes[index]
            tube_msg = TubeVisualState()
            tube_msg.index = tube.index
            tube_msg.label = f'T_{tube.index + 1}'
            tube_msg.opening = self.copy_point(tube.opening)
            tube_msg.filled = tube.filled
            tube_msg.fill_ratio = float(tube.fill_ratio)
            msg.tubes.append(tube_msg)

        return msg

    def make_delete_all_marker(self) -> Marker:
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    def make_rack_markers(self, point: Point, start_id: int) -> List[Marker]:
        size = self.get_parameter('rack_size').value
        mesh = self.get_parameter('rack_mesh').value
        marker = self.make_base_marker(start_id, 'scene_rack')
        marker.pose.position = self.copy_point(point)
        marker.pose.position.z -= size[2] / 2.0
        marker.scale.x = float(size[0])
        marker.scale.y = float(size[1])
        marker.scale.z = float(size[2])
        marker.color.r = 0.75
        marker.color.g = 0.75
        marker.color.b = 0.82
        marker.color.a = 0.55
        if mesh:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = mesh
            marker.mesh_use_embedded_materials = True
        else:
            marker.type = Marker.CUBE

        text = self.make_text_marker(start_id + 1, 'scene_labels', 'Rack', point, 0.03)
        return [marker, text]

    def make_beaker_markers(self, point: Point, start_id: int) -> List[Marker]:
        size = self.get_parameter('beaker_size').value
        mesh = self.get_parameter('beaker_mesh').value
        marker = self.make_base_marker(start_id, 'scene_beaker')
        marker.pose.position = self.copy_point(point)
        marker.pose.position.z -= size[2] / 2.0
        marker.scale.x = float(size[0])
        marker.scale.y = float(size[1])
        marker.scale.z = float(size[2])
        marker.color.r = 0.95
        marker.color.g = 0.65
        marker.color.b = 0.20
        marker.color.a = 0.35
        if mesh:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = mesh
            marker.mesh_use_embedded_materials = True
        else:
            marker.type = Marker.CYLINDER

        text = self.make_text_marker(start_id + 1, 'scene_labels', 'Beaker', point, 0.03)
        return [marker, text]

    def make_tube_markers(self, tube: TubeState, start_id: int) -> List[Marker]:
        tube_radius = float(self.get_parameter('tube_radius').value)
        tube_height = float(self.get_parameter('tube_height').value)
        tube_mesh = self.get_parameter('tube_mesh').value
        liquid_color = self.get_parameter('liquid_color').value
        active = tube.index == self.active_tube_index

        wall = self.make_base_marker(start_id, 'scene_tubes')
        wall.pose.position = self.copy_point(tube.opening)
        wall.pose.position.z -= tube_height / 2.0
        wall.color.r = 0.85 if active else 0.65
        wall.color.g = 0.90 if active else 0.78
        wall.color.b = 0.98 if active else 0.90
        wall.color.a = 0.6 if active else 0.35
        if tube_mesh:
            wall.type = Marker.MESH_RESOURCE
            wall.mesh_resource = tube_mesh
            wall.mesh_use_embedded_materials = True
            wall.scale.x = tube_radius * 2.2
            wall.scale.y = tube_radius * 2.2
            wall.scale.z = tube_height
        else:
            wall.type = Marker.CYLINDER
            wall.scale.x = tube_radius * 2.0
            wall.scale.y = tube_radius * 2.0
            wall.scale.z = tube_height

        liquid = self.make_base_marker(start_id + 1, 'scene_liquid')
        liquid.type = Marker.CYLINDER
        liquid.pose.position = self.copy_point(tube.opening)

        effective_ratio = max(0.0, min(1.0, tube.fill_ratio if tube.filled else 0.0))
        liquid_height = max(0.001, tube_height * 0.60 * effective_ratio)
        bottom_z = tube.opening.z - tube_height + (tube_height * 0.05)
        liquid.pose.position.z = bottom_z + liquid_height / 2.0
        liquid.scale.x = max(0.001, tube_radius * 1.5)
        liquid.scale.y = max(0.001, tube_radius * 1.5)
        liquid.scale.z = liquid_height
        liquid.color.r = float(liquid_color[0])
        liquid.color.g = float(liquid_color[1])
        liquid.color.b = float(liquid_color[2])
        liquid.color.a = float(liquid_color[3]) if tube.filled else 0.0

        highlight = self.make_base_marker(start_id + 2, 'scene_tube_highlight')
        highlight.type = Marker.SPHERE
        highlight.pose.position = self.copy_point(tube.opening)
        highlight.scale.x = tube_radius * 3.0
        highlight.scale.y = tube_radius * 3.0
        highlight.scale.z = tube_radius * 3.0
        highlight.color.r = 1.0 if active else 0.1
        highlight.color.g = 0.92 if active else 0.7
        highlight.color.b = 0.12 if active else 0.9
        highlight.color.a = 0.75 if active else 0.15

        label_text = f'T_{tube.index + 1}'
        if tube.filled:
            label_text += f' {int(math.ceil(effective_ratio * 100.0))}%'
        label = self.make_text_marker(start_id + 3, 'scene_labels', label_text, tube.opening, 0.02)
        return [wall, liquid, highlight, label]

    def make_base_marker(self, marker_id: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def make_text_marker(
        self,
        marker_id: int,
        namespace: str,
        text: str,
        point: Point,
        z_offset: float,
    ) -> Marker:
        marker = self.make_base_marker(marker_id, namespace)
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position = self.copy_point(point)
        marker.pose.position.z += z_offset
        marker.scale.z = 0.018
        marker.color.r = 0.95
        marker.color.g = 0.95
        marker.color.b = 0.95
        marker.color.a = 1.0
        marker.text = text
        return marker

    @staticmethod
    def copy_point(point: Point) -> Point:
        copied = Point()
        copied.x = point.x
        copied.y = point.y
        copied.z = point.z
        return copied


def main(args=None):
    rclpy.init(args=args)
    node = SceneBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
