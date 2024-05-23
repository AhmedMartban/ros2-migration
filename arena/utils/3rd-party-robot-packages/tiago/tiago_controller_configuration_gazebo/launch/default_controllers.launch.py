import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='robot_namespace',
            default_value=''
        ),
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
            default_value='mobile_base_controller'
        ),
        launch.actions.DeclareLaunchArgument(
            name='controller_list',
            default_value='mobile_base_controller'
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='default_controllers_spawner',
            output='screen',
            parameters=[
                {
                    'mobile_base_controller/base_frame_id': launch.substitutions.LaunchConfiguration('robot_namespace')
                },
                {
                    'mobile_base_controller/odom_frame_id': launch.substitutions.LaunchConfiguration('robot_namespace')
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
