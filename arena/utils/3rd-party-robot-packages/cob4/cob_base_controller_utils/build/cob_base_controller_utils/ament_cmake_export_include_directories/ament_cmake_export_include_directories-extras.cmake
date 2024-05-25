# generated from ament_cmake_export_include_directories/cmake/ament_cmake_export_include_directories-extras.cmake.in

set(_exported_include_dirs "/opt/ros/humble/include/std_srvs;/opt/ros/humble/include/rosidl_runtime_c;/opt/ros/humble/include/rosidl_typesupport_interface;/opt/ros/humble/include/rcutils;/opt/ros/humble/include/rosidl_runtime_cpp;/opt/ros/humble/include/rosidl_typesupport_fastrtps_c;/opt/ros/humble/include/rosidl_typesupport_fastrtps_cpp;/opt/ros/humble/include/rosidl_typesupport_c;/opt/ros/humble/include/rmw;/opt/ros/humble/include/rosidl_typesupport_cpp;/opt/ros/humble/include/rcpputils;/opt/ros/humble/include/rosidl_typesupport_introspection_c;/opt/ros/humble/include/rosidl_typesupport_introspection_cpp;/opt/ros/humble/include/rclcpp;/opt/ros/humble/include/ament_index_cpp;/opt/ros/humble/include/libstatistics_collector;/opt/ros/humble/include/rcl;/opt/ros/humble/include/builtin_interfaces;/opt/ros/humble/include/rosgraph_msgs;/opt/ros/humble/include/rcl_yaml_param_parser;/opt/ros/humble/include/statistics_msgs;/opt/ros/humble/include/tracetools;/opt/ros/humble/include/std_msgs;/opt/ros/humble/include/geometry_msgs;/usr/include;/opt/ros/humble/include/tf2;/opt/ros/humble/include/nav_msgs;/opt/ros/humble/include/urdf;/opt/ros/humble/include/pluginlib;/opt/ros/humble/include/urdf_parser_plugin;/opt/ros/humble/include/urdfdom;/opt/ros/humble/include/urdfdom_headers")

# append include directories to cob_base_controller_utils_INCLUDE_DIRS
# warn about not existing paths
if(NOT _exported_include_dirs STREQUAL "")
  find_package(ament_cmake_core QUIET REQUIRED)
  foreach(_exported_include_dir ${_exported_include_dirs})
    if(NOT IS_DIRECTORY "${_exported_include_dir}")
      message(WARNING "Package 'cob_base_controller_utils' exports the include directory '${_exported_include_dir}' which doesn't exist")
    endif()
    normalize_path(_exported_include_dir "${_exported_include_dir}")
    list(APPEND cob_base_controller_utils_INCLUDE_DIRS "${_exported_include_dir}")
  endforeach()
endif()
