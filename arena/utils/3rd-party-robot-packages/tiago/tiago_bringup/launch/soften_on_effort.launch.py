import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='marco_tools',
            executable='soften_arm_on_effort.py',
            name='soften_arm_on_effort',
            output='screen',
            parameters=[
                get_package_share_directory(
                    'tiago_bringup') + '/config/tiago_soften_values.yaml'
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
