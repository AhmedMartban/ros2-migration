import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='controller_manager',
            executable='controller_manager',
            name='grav_compensation_controller_loader',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='controller_manager',
            name='wrist_controller_loader',
            output='screen'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
