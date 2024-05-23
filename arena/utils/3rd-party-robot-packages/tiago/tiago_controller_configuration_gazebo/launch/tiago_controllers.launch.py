import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='public_sim',
            default_value='false'
        ),
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
            name='laser_model'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_model'
        ),
        launch.actions.DeclareLaunchArgument(
            name='use_moveit_camera',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='robot_namespace',
            default_value=''
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
                    'tiago_controller_configuration_gazebo'), 'launch/default_controllers.launch.py')
            ),
            launch_arguments={
                'robot_namespace': launch.substitutions.LaunchConfiguration('robot_namespace'),
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'base_type_suffix': launch.substitutions.LaunchConfiguration('base_type_suffix')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
