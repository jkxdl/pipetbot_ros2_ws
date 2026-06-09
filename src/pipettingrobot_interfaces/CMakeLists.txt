cmake_minimum_required(VERSION 3.5)
project(pipettingrobot_interfaces)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(action_msgs REQUIRED)

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObject.msg"
  "msg/Circle.msg"
  "msg/CirclePosition.msg"
  "msg/CollisionInfo.msg"
  "msg/TubeVisualState.msg"
  "msg/PipettingSceneState.msg"
  "msg/PipettingTaskStatus.msg"
  "srv/GetDetections.srv"
  "srv/GetCircleCoords.srv"
  "srv/SetTargetPose.srv"
  "srv/SetJointAngles.srv"
  "srv/GetExpectedCircleCount.srv"
  "srv/StartPipetting.srv"
  "srv/SetCircleCount.srv"
  "srv/SetTubeVisualState.srv"
  "srv/SetActiveTube.srv"
  "action/SetTargetPose.action"
  "action/SetGPIOCombination.action"
  "action/GetCircleCoords.action"
  "action/GetExpectedCirclePosition.action"
  DEPENDENCIES geometry_msgs
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
# Ensure that packages depending on this package also get the generated code
ament_export_dependencies(rosidl_default_runtime)

ament_package()
