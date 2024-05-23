import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='laser_model'
        ),
        launch.actions.DeclareLaunchArgument(
            name='rgbd_sensors'
        ),
        launch.actions.DeclareLaunchArgument(
            name='multi',
            default_value=''
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
