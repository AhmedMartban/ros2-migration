import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='hand_controller_spawner',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='simple_grasping_action',
            executable='simple_grasping_action',
            name='hand_grasping_action',
            output='screen',
            parameters=[
                {
                    'target_controller': 'hand_controller'
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
