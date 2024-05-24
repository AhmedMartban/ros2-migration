import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='teb_local_planner',
            executable='test_optim_node',
            name='test_optim_node',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='rviz',
            executable='rviz',
            name='rviz'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
