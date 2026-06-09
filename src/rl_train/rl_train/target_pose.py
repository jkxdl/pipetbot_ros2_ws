import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Header

class TargetPoseService(Node):
    def __init__(self):
        super().__init__('get_target_pose_server')
        # parameters
        self.declare_parameter('prefixes', ['cup', 'tube'])
        self.declare_parameter('cup_height', 0.12)
        self.declare_parameter('tube_height', 0.25)
        self.prefixes = self.get_parameter('prefixes').value
        self.cup_h = self.get_parameter('cup_height').value
        self.tube_h = self.get_parameter('tube_height').value

        # cache model_states
        self.latest_states = None
        self.create_subscription(ModelStates, '/model_states', self.cb_model_states, 10)

        # offer service
        self.srv = self.create_service(
            GetEntityState, '/get_target_pose', self.handle_get_target_pose)
        self.get_logger().info('TargetPoseService ready, providing /get_target_pose')

    def cb_model_states(self, msg: ModelStates):
        self.latest_states = msg

    def handle_get_target_pose(self, req, resp):
        name = req.name
        # validate prefix
        if not any(name.startswith(p) for p in self.prefixes):
            self.get_logger().warn(f'Invalid prefix for name: {name}')
            resp.success = False
            return resp
        # ensure we have model_states
        if self.latest_states is None:
            self.get_logger().warn('No /model_states received yet')
            resp.success = False
            return resp
        # find model index
        try:
            idx = self.latest_states.name.index(name)
        except ValueError:
            self.get_logger().warn(f'Model {name} not found in /model_states')
            resp.success = False
            return resp
        pose = self.latest_states.pose[idx]
        # build response state
        resp.state.pose = pose
        # modify z for target center
        if name.startswith('cup'):
            resp.state.pose.position.z = pose.position.z + self.cup_h/2.0
        else:
            resp.state.pose.position.z = pose.position.z + self.tube_h
        # header
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = req.reference_frame or 'world'
        resp.header = hdr  # type: ignore
        resp.success = True
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
