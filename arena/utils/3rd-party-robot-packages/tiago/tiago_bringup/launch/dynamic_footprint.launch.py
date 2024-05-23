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
            name='remap_footprint_to',
            default_value=''
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
            name='ee_suffix',
            default_value="$(eval (end_effector if end_effector else 'no-ee'))"
        ),
        launch_ros.actions.Node(
            package='dynamic_footprint',
            executable='dynamic_footprint_publisher.py',
            name='dynamic_footprint_publisher',
            parameters=[
                {
                    'tf_prefix': launch.substitutions.LaunchConfiguration('robot_namespace')
                },
                get_package_share_directory(
                    'custom_ee_configuration') + '/config/dynamic_footprint_arm_custom.yaml',
                launch.substitutions.LaunchConfiguration('ee_suffix'),
                get_package_share_directory(
                    'tiago_bringup') + '/config/dynamic_footprint_no-arm.yaml'
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
