cmake_minimum_required(VERSION 3.5)
project(brics_actuator)

find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(catkin REQUIRED)
set(CMAKE_CXX_STANDARD 1 4)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        msg/CartesianWrench.msg
        msg/CartesianPose.msg
        msg/JointConstraint.msg
        msg/CartesianTwist.msg
        msg/JointTorques.msg
        msg/CartesianVector.msg
        msg/JointAccelerations.msg
        msg/JointPositions.msg
        msg/Poison.msg
        msg/JointVelocities.msg
        msg/JointImpedances.msg
        msg/JointValue.msg
    DEPENDENCIES
        geometry_msgs
)

ament_package()
