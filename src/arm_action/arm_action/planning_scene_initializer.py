import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


class PlanningSceneInitializer(Node):
    def __init__(self):
        super().__init__('planning_scene_initializer')

        self.declare_parameter('publish_once_delay', 2.0)
        self.declare_parameter('add_ground', True)
        self.declare_parameter('add_wall', True)
        self.declare_parameter('ground_frame', 'world')
        self.declare_parameter('wall_frame', 'world')

        self._planning_scene_client = self.create_client(
            ApplyPlanningScene, '/apply_planning_scene'
        )
        while not self._planning_scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /apply_planning_scene 服务...')

        delay = float(self.get_parameter('publish_once_delay').value)
        self._timer = self.create_timer(delay, self._apply_scene_once)
        self._applied = False
        self.get_logger().info('Planning scene initializer ready.')

    def _apply_scene_once(self):
        if self._applied:
            return

        collision_objects = []

        if bool(self.get_parameter('add_ground').value):
            ground = CollisionObject()
            ground.header.frame_id = str(self.get_parameter('ground_frame').value)
            ground.id = 'ground_plane'
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [2.0, 2.0, 0.01]
            ground.primitives.append(primitive)

            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.0
            pose.position.z = -0.01
            pose.orientation.w = 1.0
            ground.primitive_poses.append(pose)
            ground.operation = CollisionObject.ADD
            collision_objects.append(ground)

        if bool(self.get_parameter('add_wall').value):
            wall = CollisionObject()
            wall.header.frame_id = str(self.get_parameter('wall_frame').value)
            wall.id = 'wall_obstacle'
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [2.0, 0.05, 2.0]
            wall.primitives.append(primitive)

            pose = Pose()
            pose.position.x = 0.0
            pose.position.y = 0.8
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            wall.primitive_poses.append(pose)
            wall.operation = CollisionObject.ADD
            collision_objects.append(wall)

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.extend(collision_objects)

        request = ApplyPlanningScene.Request()
        request.scene = planning_scene
        future = self._planning_scene_client.call_async(request)
        future.add_done_callback(self._handle_apply_response)
        self._applied = True

    def _handle_apply_response(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('规划场景障碍物已初始化。')
            else:
                self.get_logger().warn('规划场景障碍物初始化失败。')
        except Exception as exc:
            self.get_logger().error(f'应用规划场景失败: {exc}')
        finally:
            self.destroy_timer(self._timer)


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneInitializer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('节点已中断。')
    finally:
        node.destroy_node()
        rclpy.shutdown()
