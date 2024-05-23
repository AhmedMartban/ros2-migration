import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='video_device',
            default_value='/dev/video0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='framerate',
            default_value='14'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera',
            default_value='camera'
        ),
        launch.actions.DeclareLaunchArgument(
            name='rotation',
            default_value='0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='side',
            default_value='right_'
        ),
        launch.actions.DeclareLaunchArgument(
            name='side',
            default_value='left_'
        ),
        launch.actions.DeclareLaunchArgument(
            name='side',
            default_value=''
        ),
        launch_ros.actions.Node(
            package='usb_cam',
            executable='usb_cam_node',
            name=launch.substitutions.LaunchConfiguration('camera'),
            parameters=[
                {
                    'video_device': launch.substitutions.LaunchConfiguration('video_device')
                },
                {
                    'image_width': '640'
                },
                {
                    'image_height': '480'
                },
                {
                    'pixel_format': 'yuyv'
                },
                {
                    'framerate': launch.substitutions.LaunchConfiguration('framerate')
                },
                {
                    'camera_name': launch.substitutions.LaunchConfiguration('camera')
                },
                {
                    'camera_frame_id': launch.substitutions.LaunchConfiguration('side')
                },
                {
                    'camera_info_url': 'file:///tmp/cam.yaml'
                }
            ]
        ),
        launch_ros.actions.Node(
            package='tf',
            executable='static_transform_publisher',
            name=launch.substitutions.LaunchConfiguration('side')
        ),
        launch_ros.actions.Node(
            package='tf',
            executable='static_transform_publisher',
            name=launch.substitutions.LaunchConfiguration('side')
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
