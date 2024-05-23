import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='True'
        ),
        launch.actions.DeclareLaunchArgument(
            name='end_effector',
            default_value='pal-hey5'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ft_sensor',
            default_value='schunk-ft'
        ),
        launch.actions.DeclareLaunchArgument(
            name='laser_model',
            default_value='sick-571'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_model',
            default_value='orbbec-astra'
        ),
        launch.actions.DeclareLaunchArgument(
            name='wrist_model',
            default_value='wrist-2010'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='pmb2'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_controller_configuration'), 'launch/default_controllers.launch.py')
            ),
            launch_arguments={
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'base_type_suffix': launch.substitutions.LaunchConfiguration('base_type_suffix')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
