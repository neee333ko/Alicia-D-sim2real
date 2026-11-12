cmake_minimum_required(VERSION 3.8)
project(alicia_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Place.srv"
  "srv/Setup.srv"
  "action/RunOnce.action"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
