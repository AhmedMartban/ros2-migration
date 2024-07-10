cmake_minimum_required(VERSION 3.5)
project(arena_evaluation)

# Add support for C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()



find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
set(INCLUDE_DIRS ${ament_cmake_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS} ${rclpy_INCLUDE_DIRS} ${rclcpp_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRS})
set(LIBRARY_DIRS ${ament_cmake_LIBRARIES} ${sensor_msgs_LIBRARIES}
  ${nav_msgs_LIBRARIES} ${rclpy_LIBRARIES} ${rclcpp_LIBRARIES}
  ${geometry_msgs_LIBRARIES} ${std_msgs_LIBRARIES})
set(LIBS ${ament_cmake_LIBRARIES} ${sensor_msgs_LIBRARIES} ${nav_msgs_LIBRARIES}
  ${rclpy_LIBRARIES} ${rclcpp_LIBRARIES} ${geometry_msgs_LIBRARIES}
  ${std_msgs_LIBRARIES})


install(PROGRAMS scripts/data_recorder_node scripts/get_metrics
  scripts/create_plots scripts/process_data
  DESTINATION bin)

ament_export_dependencies(ament_cmake)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(rclpy)
ament_export_dependencies(rclcpp)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(std_msgs)
ament_export_include_directories(${INCLUDE_DIRS})

ament_package()
