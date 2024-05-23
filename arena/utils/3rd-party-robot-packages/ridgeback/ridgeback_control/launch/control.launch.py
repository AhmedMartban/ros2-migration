import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='controller_spawner'
        ),
        launch_ros.actions.Node(
            package='robot_localization',
            executable='ekf_localization_node',
            name='ekf_localization'
        ),
        launch_ros.actions.Node(
            package='twist_mux',
            executable='twist_mux',
            name='twist_mux',
            parameters=[
                get_package_share_directory(
                    'ridgeback_control') + '/config/twist_mux.yaml'
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
