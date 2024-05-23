import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch_ros.actions.Node(
            package='camera_calibration_files',
            executable='copy_calibration_files.sh',
            name='copy_calibration_files',
            output='screen',
            parameters=[
                {
                    '/xtion/driver/use_device_time': 'false'
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'openni2_launch'), 'launch/openni2.launch.py')
            ),
            launch_arguments={
                'camera': 'xtion',
                'publish_tf': 'false',
                'depth_registration': 'true',
                'rgb_camera_info_url': os.environ.get('HOME'),
                'depth_camera_info_url': os.environ.get('HOME')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
