# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_move_base_flex_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED move_base_flex_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(move_base_flex_FOUND FALSE)
  elseif(NOT move_base_flex_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(move_base_flex_FOUND FALSE)
  endif()
  return()
endif()
set(_move_base_flex_CONFIG_INCLUDED TRUE)

# output package information
if(NOT move_base_flex_FIND_QUIETLY)
  message(STATUS "Found move_base_flex: 0.4.0 (${move_base_flex_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'move_base_flex' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${move_base_flex_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(move_base_flex_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${move_base_flex_DIR}/${_extra}")
endforeach()
