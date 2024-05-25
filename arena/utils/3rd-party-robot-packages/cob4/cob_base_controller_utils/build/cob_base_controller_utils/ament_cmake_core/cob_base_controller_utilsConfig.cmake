# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_cob_base_controller_utils_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED cob_base_controller_utils_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(cob_base_controller_utils_FOUND FALSE)
  elseif(NOT cob_base_controller_utils_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(cob_base_controller_utils_FOUND FALSE)
  endif()
  return()
endif()
set(_cob_base_controller_utils_CONFIG_INCLUDED TRUE)

# output package information
if(NOT cob_base_controller_utils_FIND_QUIETLY)
  message(STATUS "Found cob_base_controller_utils: 0.8.18 (${cob_base_controller_utils_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'cob_base_controller_utils' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${cob_base_controller_utils_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(cob_base_controller_utils_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${cob_base_controller_utils_DIR}/${_extra}")
endforeach()
