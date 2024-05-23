import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='pmb2'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'sb04_bridge'), 'launch/mm11.launch.py')
            ),
            launch_arguments={
                'flags': "'-y -l -i -g -p'"
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
