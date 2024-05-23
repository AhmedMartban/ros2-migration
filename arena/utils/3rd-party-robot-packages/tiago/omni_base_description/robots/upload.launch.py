import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='multi',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='front_laser_model',
            default_value='sick-571'
        ),
        launch.actions.DeclareLaunchArgument(
            name='rear_laser_model',
            default_value='sick-571'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
