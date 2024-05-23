import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='controller_list',
            default_value='torso_controller               head_controller               arm_controller               arm_current_limit_controller               mobile_base_controller               wheels_current_limit_controller'
        ),
        launch.actions.DeclareLaunchArgument(
            name='controller_list',
            default_value='mobile_base_controller               wheels_current_limit_controller'
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='default_controllers_spawner',
            output='screen'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
