import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='component'
        ),
        launch.actions.DeclareLaunchArgument(
            name='chain_base_link'
        ),
        launch.actions.DeclareLaunchArgument(
            name='chain_tip_link'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
