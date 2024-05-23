import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='public_sim',
            default_value='false'
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='hand_controller_spawner',
            output='screen'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
