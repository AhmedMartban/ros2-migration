import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='joy_dev',
            default_value='$(optenv RIDGEBACK_JOY_DEVICE /dev/input/ps4)'
        ),
        launch.actions.DeclareLaunchArgument(
            name='joystick',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='config',
            default_value='planar'
        ),
        launch_ros.actions.Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('joystick'))
        ),
        launch_ros.actions.Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy',
            condition=launch.conditions.IfCondition(
                launch.substitutions.LaunchConfiguration('joystick'))
        ),
        launch_ros.actions.Node(
            package='interactive_marker_twist_server',
            executable='marker_server',
            name='twist_marker_server',
            parameters=[
                launch.substitutions.LaunchConfiguration('config')
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
