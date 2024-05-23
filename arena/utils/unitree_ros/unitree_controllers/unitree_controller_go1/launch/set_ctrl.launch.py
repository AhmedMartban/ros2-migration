import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='robot_namespace'
        ),
        launch.actions.DeclareLaunchArgument(
            name='target_state',
            default_value='-1'
        ),
        launch_ros.actions.Node(
            package='unitree_controller_go1',
            executable='unitree_controller_go1',
            name='unitree_controller_go1',
            output='screen',
            parameters=[
                {
                    'robot_namespace': launch.substitutions.LaunchConfiguration('robot_namespace')
                },
                {
                    'target_state': launch.substitutions.LaunchConfiguration('target_state')
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
