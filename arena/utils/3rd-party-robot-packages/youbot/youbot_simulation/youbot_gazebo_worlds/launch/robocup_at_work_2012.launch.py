import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='gui',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='headless',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='debug',
            default_value='false'
        ),
        launch_ros.actions.Node(
            package='gazebo_ros',
            executable='spawn_model',
            name='gazebo_world_model',
            output='screen',
            parameters=[
                {
                    'world_description': None
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'gazebo_ros'), 'launch/empty_world.launch.py')
            ),
            launch_arguments={
                'paused': 'false',
                'use_sim_time': launch.substitutions.LaunchConfiguration('use_sim_time'),
                'gui': launch.substitutions.LaunchConfiguration('gui'),
                'headless': launch.substitutions.LaunchConfiguration('headless'),
                'debug': launch.substitutions.LaunchConfiguration('debug'),
                'world_name': 'worlds/empty.world'
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
