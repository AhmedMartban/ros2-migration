import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm'
        ),
        launch.actions.DeclareLaunchArgument(
            name='end_effector'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ft_sensor'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type'
        ),
        launch.actions.DeclareLaunchArgument(
            name='file_suffix',
            default_value="$(eval (('no-arm' if not arm else (end_effector if end_effector else 'no-ee') + ('_' +  ft_sensor if ft_sensor else ''))))"
        ),
        launch_ros.actions.Node(
            package='play_motion',
            executable='play_motion',
            name='play_motion',
            parameters=[
                launch.substitutions.LaunchConfiguration('file_suffix'),
                get_package_share_directory(
                    'custom_ee_configuration') + '/config/approach_planner/approach_planner.yaml'
            ]
        ),
        launch_ros.actions.Node(
            package='play_motion',
            executable='is_already_there.py',
            name='is_already_there'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
