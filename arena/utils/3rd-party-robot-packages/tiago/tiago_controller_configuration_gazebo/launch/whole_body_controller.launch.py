import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='controller_manager',
            executable='controller_manager',
            name='controller_manager_wbc_loader',
            output='screen'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_wbc'), 'launch/tiago_capsule_description.launch.py')
            )
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
