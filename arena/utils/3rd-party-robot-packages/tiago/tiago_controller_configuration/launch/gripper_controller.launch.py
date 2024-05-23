import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='gripper_controllers_spawner',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='tiago_controller_configuration',
            executable='home_gripper.py',
            name='home_gripper',
            output='screen'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_gripper_controller_configuration'), 'launch/current_limit_controllers.launch.py')
            )
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_parallel_gripper_wrapper'), 'launch/parallel_gripper.launch.py')
            )
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
