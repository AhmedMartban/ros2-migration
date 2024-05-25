// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cob_srvs:srv/SetFloat.idl
// generated code does not contain a copyright notice

#ifndef COB_SRVS__SRV__DETAIL__SET_FLOAT__FUNCTIONS_H_
#define COB_SRVS__SRV__DETAIL__SET_FLOAT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cob_srvs/msg/rosidl_generator_c__visibility_control.h"

#include "cob_srvs/srv/detail/set_float__struct.h"

/// Initialize srv/SetFloat message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cob_srvs__srv__SetFloat_Request
 * )) before or use
 * cob_srvs__srv__SetFloat_Request__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__init(cob_srvs__srv__SetFloat_Request * msg);

/// Finalize srv/SetFloat message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Request__fini(cob_srvs__srv__SetFloat_Request * msg);

/// Create srv/SetFloat message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cob_srvs__srv__SetFloat_Request__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
cob_srvs__srv__SetFloat_Request *
cob_srvs__srv__SetFloat_Request__create();

/// Destroy srv/SetFloat message.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Request__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Request__destroy(cob_srvs__srv__SetFloat_Request * msg);

/// Check for srv/SetFloat message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__are_equal(const cob_srvs__srv__SetFloat_Request * lhs, const cob_srvs__srv__SetFloat_Request * rhs);

/// Copy a srv/SetFloat message.
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
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__copy(
  const cob_srvs__srv__SetFloat_Request * input,
  cob_srvs__srv__SetFloat_Request * output);

/// Initialize array of srv/SetFloat messages.
/**
 * It allocates the memory for the number of elements and calls
 * cob_srvs__srv__SetFloat_Request__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__Sequence__init(cob_srvs__srv__SetFloat_Request__Sequence * array, size_t size);

/// Finalize array of srv/SetFloat messages.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Request__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Request__Sequence__fini(cob_srvs__srv__SetFloat_Request__Sequence * array);

/// Create array of srv/SetFloat messages.
/**
 * It allocates the memory for the array and calls
 * cob_srvs__srv__SetFloat_Request__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
cob_srvs__srv__SetFloat_Request__Sequence *
cob_srvs__srv__SetFloat_Request__Sequence__create(size_t size);

/// Destroy array of srv/SetFloat messages.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Request__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Request__Sequence__destroy(cob_srvs__srv__SetFloat_Request__Sequence * array);

/// Check for srv/SetFloat message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__Sequence__are_equal(const cob_srvs__srv__SetFloat_Request__Sequence * lhs, const cob_srvs__srv__SetFloat_Request__Sequence * rhs);

/// Copy an array of srv/SetFloat messages.
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
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Request__Sequence__copy(
  const cob_srvs__srv__SetFloat_Request__Sequence * input,
  cob_srvs__srv__SetFloat_Request__Sequence * output);

/// Initialize srv/SetFloat message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cob_srvs__srv__SetFloat_Response
 * )) before or use
 * cob_srvs__srv__SetFloat_Response__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__init(cob_srvs__srv__SetFloat_Response * msg);

/// Finalize srv/SetFloat message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Response__fini(cob_srvs__srv__SetFloat_Response * msg);

/// Create srv/SetFloat message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cob_srvs__srv__SetFloat_Response__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
cob_srvs__srv__SetFloat_Response *
cob_srvs__srv__SetFloat_Response__create();

/// Destroy srv/SetFloat message.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Response__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Response__destroy(cob_srvs__srv__SetFloat_Response * msg);

/// Check for srv/SetFloat message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__are_equal(const cob_srvs__srv__SetFloat_Response * lhs, const cob_srvs__srv__SetFloat_Response * rhs);

/// Copy a srv/SetFloat message.
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
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__copy(
  const cob_srvs__srv__SetFloat_Response * input,
  cob_srvs__srv__SetFloat_Response * output);

/// Initialize array of srv/SetFloat messages.
/**
 * It allocates the memory for the number of elements and calls
 * cob_srvs__srv__SetFloat_Response__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__Sequence__init(cob_srvs__srv__SetFloat_Response__Sequence * array, size_t size);

/// Finalize array of srv/SetFloat messages.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Response__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Response__Sequence__fini(cob_srvs__srv__SetFloat_Response__Sequence * array);

/// Create array of srv/SetFloat messages.
/**
 * It allocates the memory for the array and calls
 * cob_srvs__srv__SetFloat_Response__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
cob_srvs__srv__SetFloat_Response__Sequence *
cob_srvs__srv__SetFloat_Response__Sequence__create(size_t size);

/// Destroy array of srv/SetFloat messages.
/**
 * It calls
 * cob_srvs__srv__SetFloat_Response__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
void
cob_srvs__srv__SetFloat_Response__Sequence__destroy(cob_srvs__srv__SetFloat_Response__Sequence * array);

/// Check for srv/SetFloat message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__Sequence__are_equal(const cob_srvs__srv__SetFloat_Response__Sequence * lhs, const cob_srvs__srv__SetFloat_Response__Sequence * rhs);

/// Copy an array of srv/SetFloat messages.
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
ROSIDL_GENERATOR_C_PUBLIC_cob_srvs
bool
cob_srvs__srv__SetFloat_Response__Sequence__copy(
  const cob_srvs__srv__SetFloat_Response__Sequence * input,
  cob_srvs__srv__SetFloat_Response__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // COB_SRVS__SRV__DETAIL__SET_FLOAT__FUNCTIONS_H_
