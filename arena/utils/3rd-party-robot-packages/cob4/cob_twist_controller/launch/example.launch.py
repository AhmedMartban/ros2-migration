import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='topic_tools',
            executable='relay',
            name='joint_states_relay',
            output='screen',
            parameters=[
                {
                    'robot_description': None
                }
            ]
        ),
        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {
                    'robot_description': None
                },
                {
                    'publish_frequency': '50.0'
                },
                {
                    'tf_prefix': ''
                }
            ]
        ),
        launch_ros.actions.Node(
            package='cob_twist_controller',
            executable='cob_twist_controller_node',
            name='twist_controller',
            output='screen',
            parameters=[
                {
                    'robot_description': None
                }
            ]
        ),
        launch_ros.actions.Node(
            package='cob_frame_tracker',
            executable='cob_frame_tracker_node',
            name='frame_tracker',
            output='screen',
            parameters=[
                {
                    'robot_description': None
                }
            ]
        ),
        launch_ros.actions.Node(
            package='cob_frame_tracker',
            executable='interactive_frame_target_node',
            name='interactive_target',
            output='screen',
            parameters=[
                {
                    'robot_description': None
                }
            ]
        ),
        launch_ros.actions.Node(
            package='cob_twist_controller',
            executable='debug_trajectory_marker_node',
            name='debug_trajectory_marker_node',
            output='screen',
            parameters=[
                {
                    'robot_description': None
                },
                {
                    'target_frame': 'arm_7_target'
                }
            ]
        ),
        launch_ros.actions.Node(
            package='rviz',
            executable='rviz',
            name='rviz',
            parameters=[
                {
                    'robot_description': None
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
