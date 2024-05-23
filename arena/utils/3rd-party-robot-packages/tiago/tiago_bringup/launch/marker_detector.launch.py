import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='markerId',
            default_value='582'
        ),
        launch.actions.DeclareLaunchArgument(
            name='markerSize',
            default_value='0.09'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_frame',
            default_value='/xtion_rgb_optical_frame'
        ),
        launch.actions.DeclareLaunchArgument(
            name='marker_frame',
            default_value='/aruco_marker_frame'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ref_frame',
            default_value=''
        ),
        launch_ros.actions.Node(
            package='aruco_ros',
            executable='single',
            name='aruco_single',
            parameters=[
                {
                    'image_is_rectified': 'True'
                },
                {
                    'marker_size': launch.substitutions.LaunchConfiguration('markerSize')
                },
                {
                    'marker_id': launch.substitutions.LaunchConfiguration('markerId')
                },
                {
                    'reference_frame': launch.substitutions.LaunchConfiguration('ref_frame')
                },
                {
                    'camera_frame': launch.substitutions.LaunchConfiguration('camera_frame')
                },
                {
                    'marker_frame': launch.substitutions.LaunchConfiguration('marker_frame')
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
