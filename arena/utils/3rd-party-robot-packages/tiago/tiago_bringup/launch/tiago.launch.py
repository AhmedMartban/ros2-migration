import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='end_effector',
            default_value='pal-hey5'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ft_sensor',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='laser_model',
            default_value='sick-571'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_model',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='wrist_model',
            default_value='wrist-2010'
        ),
        launch.actions.DeclareLaunchArgument(
            name='no_safety_eps',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='omni_base'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value='_omni'
        ),
        launch.actions.DeclareLaunchArgument(
            name='calibrated_multipliers_dir',
            default_value=os.environ.get(
                'pal_wheel_calibration_calibration_path')
        ),
        launch.actions.DeclareLaunchArgument(
            name='multipliers_dir',
            default_value="$(eval calibrated_multipliers_dir if (calibrated_multipliers_dir != '') else find('tiago_controller_configuration') + '/config')"
        ),
        launch.actions.DeclareLaunchArgument(
            name='file_suffix',
            default_value="$(eval ('no-arm' if not arm else ((end_effector if end_effector else 'no-ee') + ('_' +  ft_sensor if ft_sensor else ''))))"
        ),
        launch_ros.actions.Node(
            package='gazebo_utils',
            executable='odom.py',
            name='odom_pub',
            parameters=[
                {
                    'base_frame': 'base_footprint'
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_description'), 'robots/upload.launch.py')
            ),
            launch_arguments={
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'end_effector': launch.substitutions.LaunchConfiguration('end_effector'),
                'ft_sensor': launch.substitutions.LaunchConfiguration('ft_sensor'),
                'laser_model': launch.substitutions.LaunchConfiguration('laser_model'),
                'camera_model': launch.substitutions.LaunchConfiguration('camera_model'),
                'wrist_model': launch.substitutions.LaunchConfiguration('wrist_model'),
                'no_safety_eps': launch.substitutions.LaunchConfiguration('no_safety_eps'),
                'base_type': launch.substitutions.LaunchConfiguration('base_type')
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_bringup'), 'launch/dynamixel_node.launch.py')
            )
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_bringup'), 'launch/tiago_bringup.launch.py')
            ),
            launch_arguments={
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'end_effector': launch.substitutions.LaunchConfiguration('end_effector'),
                'ft_sensor': launch.substitutions.LaunchConfiguration('ft_sensor'),
                'laser_model': launch.substitutions.LaunchConfiguration('laser_model'),
                'camera_model': launch.substitutions.LaunchConfiguration('camera_model'),
                'wrist_model': launch.substitutions.LaunchConfiguration('wrist_model'),
                'base_type_suffix': launch.substitutions.LaunchConfiguration('base_type_suffix'),
                'base_type': launch.substitutions.LaunchConfiguration('base_type')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
