import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm_name',
            default_value='arm_1'
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name=launch.substitutions.LaunchConfiguration('arm_name'),
            output='screen'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
