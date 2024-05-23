import os
import sys

import launch
import launch_ros.actions


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
            name='laser_model'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_model'
        ),
        launch.actions.DeclareLaunchArgument(
            name='gzpose',
            default_value='-x 0 -y 0 -z 0.0 -R 0.0 -P 0.0 -Y 0.0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='multi',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='model_name',
            default_value='tiago'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='pmb2'
        ),
        launch.actions.DeclareLaunchArgument(
            name='file_suffix',
            default_value="$(eval (('no-arm' if not arm else (end_effector if end_effector else 'no-ee') + ('_' +  ft_sensor if ft_sensor else ''))))"
        ),
        launch.actions.DeclareLaunchArgument(
            name='ee_suffix',
            default_value="$(eval (end_effector if end_effector else 'no-ee'))"
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
