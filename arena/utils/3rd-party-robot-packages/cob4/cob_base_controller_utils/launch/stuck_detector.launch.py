import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='timeout',
            default_value='1.0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='threshold',
            default_value='1.57'
        ),
        launch.actions.DeclareLaunchArgument(
            name='shutdown',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='component_name',
            default_value='base'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
