import rclpy
from rclpy.node import Node
import numpy as np
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class RayVisualizer(Node):
    def __init__(self):
        super().__init__('ray_visualizer')
        
        # 参数配置（须与主节点一致）
        self.ray_count = 36
        self.plane = "x>0"
        self.base_frame = "base_link"
        self.ray_length = 1.0  # 在 Rviz 中显示的射线长度（米）

        # 初始化发布器
        self.marker_pub = self.create_publisher(Marker, '/visual_rays', 10)
        
        # 生成射线方向
        self.rays = self._make_fibonacci_rays_quarter(self.ray_count, self.plane)
        
        # 定时发布 Marker
        self.timer = self.create_timer(1.0, self.publish_rays)
        self.get_logger().info(f"可视化节点启动：正在 {self.base_frame} 坐标系下绘制 {self.ray_count} 条射线...")

    def _make_fibonacci_rays_quarter(self, ray_count, plane):
        """完全对齐主节点的射线生成逻辑"""
        n_try = max(ray_count * 3, 128)
        i = np.arange(0, n_try, dtype=np.float64) + 0.5
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * i / phi
        z = 1.0 - 2.0 * i / n_try
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        x, y = r * np.cos(theta), r * np.sin(theta)
        dirs = np.stack([x, y, z], axis=-1)

        if plane == "x>0": mask = (dirs[:, 0] >= 0.0) & (dirs[:, 2] >= 0.0)
        elif plane == "y>0": mask = (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
        elif plane == "xy>0": mask = (dirs[:, 0] >= 0.0) & (dirs[:, 1] >= 0.0) & (dirs[:, 2] >= 0.0)
        else: mask = np.ones(len(dirs), dtype=bool)

        dirs = dirs[mask][:ray_count]
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        return (dirs / (norms + 1e-8)).astype(np.float32)

    def publish_rays(self):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "rays"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # 射线粗细
        marker.scale.x = 0.005 
        
        # 射线颜色 (红色)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)

        # 设置坐标点
        for ray in self.rays:
            # 起点：base_link 原点 (0,0,0)
            start_p = Point(x=0.0, y=0.0, z=0.0)
            # 终点：方向向量 * 长度
            end_p = Point(
                x=float(ray[0] * self.ray_length),
                y=float(ray[1] * self.ray_length),
                z=float(ray[2] * self.ray_length)
            )
            marker.points.append(start_p)
            marker.points.append(end_p)

        self.marker_pub.publish(marker)

def main():
    rclpy.init()
    node = RayVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()