import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='ns',
            default_value='look_to_link'
        ),
        launch.actions.DeclareLaunchArgument(
            name='link_to_look_at',
            default_value='/arm_7_link'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_frame',
            default_value='/base_link'
        ),
        launch.actions.DeclareLaunchArgument(
            name='send_goals_to_head_action',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='pointing_frame',
            default_value='head_2_link'
        ),
        launch.actions.DeclareLaunchArgument(
            name='publish_only_when_moving',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='min_distance_moved',
            default_value='0.005'
        ),
        launch.actions.DeclareLaunchArgument(
            name='check_rate',
            default_value='10'
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
