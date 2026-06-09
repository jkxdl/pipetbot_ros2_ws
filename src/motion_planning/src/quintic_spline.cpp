#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "moveit_msgs/msg/planning_scene.hpp"
#include "moveit_msgs/msg/collision_object.hpp"
#include "moveit_msgs/msg/robot_state.hpp"
#include "moveit_msgs/msg/display_trajectory.hpp"
#include "moveit_msgs/msg/position_constraint.hpp"
#include "moveit_msgs/msg/orientation_constraint.hpp"
#include "moveit_msgs/msg/constraints.hpp"
#include "moveit_msgs/msg/joint_constraint.hpp"
#include "shape_msgs/msg/solid_primitive.hpp"

// 自定义动作接口（确保先生成该接口文件）
#include "moveit_msgs/srv/apply_planning_scene.hpp"
#include "pipettingrobot_interfaces/action/set_target_pose.hpp"

#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;
using SetTargetPose = pipettingrobot_interfaces::action::SetTargetPose;

class QuinticSpline : public rclcpp::Node
{
public:
  QuinticSpline() : Node("quintic_spline")
  {
    // 规划组中各关节名称（请根据实际机器人修改）
    planning_group_joint_names_ = {
      "shoulder_joint",
      "upperArm_joint",
      "foreArm_joint",
      "wrist1_joint",
      "wrist2_joint",
      "wrist3_joint"
    };

    // 订阅 /joint_states，获取当前关节状态
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      std::bind(&QuinticSpline::jointStatesCallback, this, std::placeholders::_1));
    RCLCPP_INFO(this->get_logger(), "Subscribed to /joint_states, waiting for joint state updates...");

    // 创建 DisplayTrajectory 发布者
    display_traj_pub_ = this->create_publisher<moveit_msgs::msg::DisplayTrajectory>("display_planned_path", 10);

    // 创建 /apply_planning_scene 服务客户端用于添加障碍物
    planning_scene_client_ = this->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    while (!planning_scene_client_->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for /apply_planning_scene service...");
    }

    // 添加障碍物（例如地面和墙面，保持原设置）
    addObstacles();
    // 声明服务客户端
    rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr planning_scene_client_;

    // 创建自定义动作服务：SetTargetPose
    action_server_ = rclcpp_action::create_server<SetTargetPose>(
      this,
      "set_target_pose",
      std::bind(&QuinticSpline::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&QuinticSpline::handle_cancel, this, std::placeholders::_1),
      std::bind(&QuinticSpline::handle_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Action server [set_target_pose] created, waiting for goals...");
  }

private:
  // --- 数据成员 ---
  std::vector<std::string> planning_group_joint_names_;
  std::mutex joint_state_mutex_;
  std::map<std::string, double> current_joint_state_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Publisher<moveit_msgs::msg::DisplayTrajectory>::SharedPtr display_traj_pub_;
  rclcpp::Client<moveit_msgs::srv::ApplyPlanningScene>::SharedPtr planning_scene_client_;
  rclcpp_action::Server<SetTargetPose>::SharedPtr action_server_;

  // --- 回调函数 ---
  void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    // 只提取规划组中的关节信息
    for (size_t i = 0; i < msg->name.size(); i++) {
      if (std::find(planning_group_joint_names_.begin(), planning_group_joint_names_.end(), msg->name[i]) != planning_group_joint_names_.end()) {
        current_joint_state_[msg->name[i]] = msg->position[i];
      }
    }
  }

  // 添加障碍物（地面和墙面）到规划场景
  void addObstacles()
  {
    RCLCPP_INFO(this->get_logger(), "Adding obstacles to the planning scene...");
    auto request = std::make_shared<moveit_msgs::srv::ApplyPlanningScene::Request>();
    moveit_msgs::msg::PlanningScene planning_scene;
    planning_scene.is_diff = true;

    // 地面
    moveit_msgs::msg::CollisionObject ground;
    ground.header.frame_id = "world";
    ground.id = "ground_plane";

    shape_msgs::msg::SolidPrimitive ground_primitive;
    ground_primitive.type = ground_primitive.BOX;
    ground_primitive.dimensions = {2.0, 2.0, 0.01};
    ground.primitives.push_back(ground_primitive);

    geometry_msgs::msg::Pose ground_pose;
    ground_pose.position.x = 0.0;
    ground_pose.position.y = 0.0;
    ground_pose.position.z = -0.01;
    ground_pose.orientation.w = 1.0;
    ground.primitive_poses.push_back(ground_pose);

    ground.operation = ground.ADD;
    planning_scene.world.collision_objects.push_back(ground);

    // 墙面
    moveit_msgs::msg::CollisionObject wall;
    wall.header.frame_id = "world";
    wall.id = "wall_obstacle";

    shape_msgs::msg::SolidPrimitive wall_primitive;
    wall_primitive.type = wall_primitive.BOX;
    wall_primitive.dimensions = {2.0, 0.05, 2.0};
    wall.primitives.push_back(wall_primitive);

    geometry_msgs::msg::Pose wall_pose;
    wall_pose.position.x = 0.0;
    wall_pose.position.y = 0.5;
    wall_pose.position.z = 0.0;
    wall_pose.orientation.w = 1.0;
    wall.primitive_poses.push_back(wall_pose);

    wall.operation = wall.ADD;
    planning_scene.world.collision_objects.push_back(wall);

    request->scene = planning_scene;
    // 异步调用服务
    auto future_result = planning_scene_client_->async_send_request(request);
    RCLCPP_INFO(this->get_logger(), "Obstacles add request sent.");
  }

  // --- Action Server 回调 ---

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const SetTargetPose::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received new SetTargetPose goal.");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SetTargetPose>>)
  {
    RCLCPP_INFO(this->get_logger(), "Canceling goal request.");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SetTargetPose>> goal_handle)
  {
    // 单独线程执行
    std::thread{std::bind(&QuinticSpline::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  // --- 执行回调 ---
  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<SetTargetPose>> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing goal...");
    auto feedback = std::make_shared<SetTargetPose::Feedback>();
    auto result = std::make_shared<SetTargetPose::Result>();

    // 取出客户端传入的目标位姿
    geometry_msgs::msg::Pose target_pose = goal_handle->get_goal()->pose;
    RCLCPP_INFO(this->get_logger(), "Received target pose.");

    // 【步骤1】根据目标位姿计算 IK 得到目标关节角度（该函数中可加入约束考虑）
    std::vector<double> target_joint_angles;
    if (!computeIK(target_pose, target_joint_angles)) {
      result->success = false;
      result->message = "IK failed to produce a valid solution.";
      goal_handle->abort(result);
      return;
    }
    RCLCPP_INFO(this->get_logger(), "IK solution computed.");

    // 【步骤2】获取当前关节状态作为起始配置
    std::vector<double> start_joint_angles;
    {
      std::lock_guard<std::mutex> lock(joint_state_mutex_);
      for (auto & joint_name : planning_group_joint_names_) {
        if (current_joint_state_.find(joint_name) != current_joint_state_.end()) {
          start_joint_angles.push_back(current_joint_state_[joint_name]);
        } else {
          start_joint_angles.push_back(0.0);
        }
      }
    }

    // 【步骤3】利用五次样条对各关节从起点到终点的角度做平滑轨迹插值
    double T = 5.0;      // 总运动时间（秒）
    int N = 100;         // 采样点数
    std::vector<std::vector<double>> trajectory;
    generateQuinticTrajectory(start_joint_angles, target_joint_angles, T, N, trajectory);
    RCLCPP_INFO(this->get_logger(), "Trajectory generated via quintic spline interpolation.");

    // 选做：通过 DisplayTrajectory 消息将规划路径发布到 RViz（这里只发布第一和最后点信息）
    publishTrajectory(trajectory);

    // 模拟轨迹执行（实际应用中应将采样轨迹发送给控制器）
    rclcpp::Rate rate(1);
    for (size_t i = 0; i < trajectory.size(); ++i) {
      feedback->feedback = "Executing trajectory point " + std::to_string(i + 1) + " / " + std::to_string(trajectory.size());
      goal_handle->publish_feedback(feedback);
      rate.sleep();
    }

    result->success = true;
    result->message = "Successfully reached target pose.";
    goal_handle->succeed(result);
  }

  // --- IK 计算函数 ---  
  // 此处给出一个简单的示例 IK 函数，实际场景中应调用相应的逆运动学求解器，
  // 该示例简单地将当前关节角加上 0.1 作为伪解，且已满足约束（示例中不作复杂运算）
  bool computeIK(const geometry_msgs::msg::Pose & target_pose, std::vector<double> & joint_angles)
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    if (current_joint_state_.empty()) {
      return false;
    }
    joint_angles.clear();
    for (auto & joint_name : planning_group_joint_names_) {
      double current_angle = current_joint_state_[joint_name];
      double computed_angle = current_angle + 0.1;  // 仅示例，实际应使用机器人运动学模型
      joint_angles.push_back(computed_angle);
    }
    return true;
  }

  // --- 五次样条轨迹生成 ---  
  // 对每个关节，假定轨迹 q(t) = q0 + a3*t^3 + a4*t^4 + a5*t^5，
  // 系数 a3、a4、a5 分别由公式 a3 = 10*(qf-q0)/T^3, a4 = -15*(qf-q0)/T^4, a5 = 6*(qf-q0)/T^5 给出
  void generateQuinticTrajectory(
    const std::vector<double> & start_angles,
    const std::vector<double> & target_angles,
    double T, int N,
    std::vector<std::vector<double>> & trajectory)
  {
    size_t dof = start_angles.size();
    trajectory.clear();
    trajectory.resize(N + 1, std::vector<double>(dof, 0.0));

    std::vector<double> a3(dof), a4(dof), a5(dof);
    for (size_t i = 0; i < dof; i++) {
      double delta = target_angles[i] - start_angles[i];
      a3[i] = 10.0 * delta / std::pow(T, 3);
      a4[i] = -15.0 * delta / std::pow(T, 4);
      a5[i] = 6.0 * delta / std::pow(T, 5);
    }

    for (int i = 0; i <= N; i++) {
      double t = T * i / N;
      std::vector<double> point(dof, 0.0);
      for (size_t j = 0; j < dof; j++) {
        point[j] = start_angles[j] + a3[j] * std::pow(t, 3)
                   + a4[j] * std::pow(t, 4) + a5[j] * std::pow(t, 5);
      }
      trajectory[i] = point;
    }
  }

  // --- 可选：将轨迹发布至 RViz 以便可视化 ---
  void publishTrajectory(const std::vector<std::vector<double>> & trajectory)
  {
    moveit_msgs::msg::DisplayTrajectory display_traj_msg;

    // 构造起始状态消息
    moveit_msgs::msg::RobotState robot_state;
    sensor_msgs::msg::JointState joint_state_msg;
    {
      std::lock_guard<std::mutex> lock(joint_state_mutex_);
      for (auto & joint_name : planning_group_joint_names_) {
        joint_state_msg.name.push_back(joint_name);
        joint_state_msg.position.push_back(current_joint_state_[joint_name]);
      }
    }
    robot_state.joint_state = joint_state_msg;
    display_traj_msg.trajectory_start = robot_state;

    // 此处仅打印出采样的第一点和最后一点进行展示
    RCLCPP_INFO(this->get_logger(), "Trajectory start point:");
    for (auto val : trajectory.front()) {
      RCLCPP_INFO(this->get_logger(), "%f", val);
    }
    RCLCPP_INFO(this->get_logger(), "Trajectory end point:");
    for (auto val : trajectory.back()) {
      RCLCPP_INFO(this->get_logger(), "%f", val);
    }
    display_traj_pub_->publish(display_traj_msg);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<QuinticSpline>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
