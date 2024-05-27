// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from plan_msgs:msg/RobotStateStamped.idl
// generated code does not contain a copyright notice

#ifndef PLAN_MSGS__MSG__DETAIL__ROBOT_STATE_STAMPED__FUNCTIONS_H_
#define PLAN_MSGS__MSG__DETAIL__ROBOT_STATE_STAMPED__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "plan_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "plan_msgs/msg/detail/robot_state_stamped__struct.h"

/// Initialize msg/RobotStateStamped message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * plan_msgs__msg__RobotStateStamped
 * )) before or use
 * plan_msgs__msg__RobotStateStamped__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__init(plan_msgs__msg__RobotStateStamped * msg);

/// Finalize msg/RobotStateStamped message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
void
plan_msgs__msg__RobotStateStamped__fini(plan_msgs__msg__RobotStateStamped * msg);

/// Create msg/RobotStateStamped message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * plan_msgs__msg__RobotStateStamped__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
plan_msgs__msg__RobotStateStamped *
plan_msgs__msg__RobotStateStamped__create();

/// Destroy msg/RobotStateStamped message.
/**
 * It calls
 * plan_msgs__msg__RobotStateStamped__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
void
plan_msgs__msg__RobotStateStamped__destroy(plan_msgs__msg__RobotStateStamped * msg);

/// Check for msg/RobotStateStamped message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__are_equal(const plan_msgs__msg__RobotStateStamped * lhs, const plan_msgs__msg__RobotStateStamped * rhs);

/// Copy a msg/RobotStateStamped message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__copy(
  const plan_msgs__msg__RobotStateStamped * input,
  plan_msgs__msg__RobotStateStamped * output);

/// Initialize array of msg/RobotStateStamped messages.
/**
 * It allocates the memory for the number of elements and calls
 * plan_msgs__msg__RobotStateStamped__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__Sequence__init(plan_msgs__msg__RobotStateStamped__Sequence * array, size_t size);

/// Finalize array of msg/RobotStateStamped messages.
/**
 * It calls
 * plan_msgs__msg__RobotStateStamped__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
void
plan_msgs__msg__RobotStateStamped__Sequence__fini(plan_msgs__msg__RobotStateStamped__Sequence * array);

/// Create array of msg/RobotStateStamped messages.
/**
 * It allocates the memory for the array and calls
 * plan_msgs__msg__RobotStateStamped__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
plan_msgs__msg__RobotStateStamped__Sequence *
plan_msgs__msg__RobotStateStamped__Sequence__create(size_t size);

/// Destroy array of msg/RobotStateStamped messages.
/**
 * It calls
 * plan_msgs__msg__RobotStateStamped__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
void
plan_msgs__msg__RobotStateStamped__Sequence__destroy(plan_msgs__msg__RobotStateStamped__Sequence * array);

/// Check for msg/RobotStateStamped message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__Sequence__are_equal(const plan_msgs__msg__RobotStateStamped__Sequence * lhs, const plan_msgs__msg__RobotStateStamped__Sequence * rhs);

/// Copy an array of msg/RobotStateStamped messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_plan_msgs
bool
plan_msgs__msg__RobotStateStamped__Sequence__copy(
  const plan_msgs__msg__RobotStateStamped__Sequence * input,
  plan_msgs__msg__RobotStateStamped__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // PLAN_MSGS__MSG__DETAIL__ROBOT_STATE_STAMPED__FUNCTIONS_H_
