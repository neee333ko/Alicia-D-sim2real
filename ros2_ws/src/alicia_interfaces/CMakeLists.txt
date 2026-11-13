cmake_minimum_required(VERSION 3.8)
project(alicia_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
# ObjectPose.msg 里用了 std_msgs/Header 和 geometry_msgs/Point
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectPose.msg"
  "srv/Place.srv"
  "srv/Setup.srv"
  "action/RunOnce.action"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
