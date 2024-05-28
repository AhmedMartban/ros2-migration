# generated from ament_cmake_export_include_directories/cmake/ament_cmake_export_include_directories-extras.cmake.in

set(_exported_include_dirs "${unitree_legged_msgs_DIR}/../../../include/unitree_legged_msgs;/opt/ros/humble/include/rcutils;/opt/ros/humble/include/rcpputils;/opt/ros/humble/include/rosidl_runtime_c;/opt/ros/humble/include/rosidl_typesupport_interface;/opt/ros/humble/include/rosidl_typesupport_c;/opt/ros/humble/include/rosidl_typesupport_cpp;/opt/ros/humble/include/rosidl_typesupport_fastrtps_c;/opt/ros/humble/include/rosidl_typesupport_introspection_c;/opt/ros/humble/include/rmw;/opt/ros/humble/include/rosidl_typesupport_fastrtps_cpp;/opt/ros/humble/include/rosidl_runtime_cpp;/opt/ros/humble/include/rosidl_typesupport_introspection_cpp;/opt/ros/humble/include/std_msgs;/opt/ros/humble/include/geometry_msgs;/opt/ros/humble/include/builtin_interfaces;/opt/ros/humble/include/sensor_msgs")

# append include directories to unitree_legged_msgs_INCLUDE_DIRS
# warn about not existing paths
if(NOT _exported_include_dirs STREQUAL "")
  find_package(ament_cmake_core QUIET REQUIRED)
  foreach(_exported_include_dir ${_exported_include_dirs})
    if(NOT IS_DIRECTORY "${_exported_include_dir}")
      message(WARNING "Package 'unitree_legged_msgs' exports the include directory '${_exported_include_dir}' which doesn't exist")
    endif()
    normalize_path(_exported_include_dir "${_exported_include_dir}")
    list(APPEND unitree_legged_msgs_INCLUDE_DIRS "${_exported_include_dir}")
  endforeach()
endif()
