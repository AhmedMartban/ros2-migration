import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='tf',
            executable='static_transform_publisher',
            name='ira_static_broadcaster1'
        ),
        launch_ros.actions.Node(
            package='tf',
            executable='static_transform_publisher',
            name='ira_static_broadcaster2'
        ),
        launch_ros.actions.Node(
            package='ira_laser_tools',
            executable='laserscan_virtualizer',
            name='laserscan_virtualizer',
            output='screen',
            parameters=[
                {
                    'cloud_topic': '/cloud_in'
                },
                {
                    'base_frame': '/laser_frame'
                },
                {
                    'output_laser_topic': '/scan'
                },
                {
                    'virtual_laser_scan': 'scansx scandx'
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
