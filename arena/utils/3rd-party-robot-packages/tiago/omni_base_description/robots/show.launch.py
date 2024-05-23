import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='front_laser_model',
            default_value='sick-571'
        ),
        launch.actions.DeclareLaunchArgument(
            name='rear_laser_model',
            default_value='sick-571'
        ),
        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher'
        ),
        launch_ros.actions.Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),
        launch_ros.actions.Node(
            package='rviz',
            executable='rviz',
            name='rviz'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'omni_base_description'), 'robots/upload.launch.py')
            ),
            launch_arguments={
                'front_laser_model': launch.substitutions.LaunchConfiguration('front_laser_model'),
                'rear_laser_model': launch.substitutions.LaunchConfiguration('rear_laser_model')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
