import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='cmd_vel_out',
            default_value='mobile_base_controller/cmd_vel'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value=''
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'twist_mux'), 'launch/twist_mux.launch.py')
            ),
            launch_arguments={
                'config_locks': get_package_share_directory('tiago_bringup') + '/config/twist_mux/twist_mux_locks.yaml',
                'config_topics': get_package_share_directory('tiago_bringup') + '/config/twist_mux/twist_mux_topics.yaml',
                'config_joy': launch.substitutions.LaunchConfiguration('base_type_suffix'),
                'cmd_vel_out': launch.substitutions.LaunchConfiguration('cmd_vel_out')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
