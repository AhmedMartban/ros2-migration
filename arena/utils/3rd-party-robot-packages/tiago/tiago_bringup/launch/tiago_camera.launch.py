import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='camera_model',
            default_value='orbbec-astra'
        ),
        launch.actions.DeclareLaunchArgument(
            name='is_xtion',
            default_value="$(eval {'orbbec-astra-pro': 'false'}.get(arg('camera_model'), 'true'))"
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
