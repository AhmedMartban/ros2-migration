import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='world',
            default_value='empty'
        ),
        launch.actions.DeclareLaunchArgument(
            name='robot',
            default_value='titanium'
        ),
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='end_effector',
            default_value='false'
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
            name='multi',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='gzpose',
            default_value='-x 0.0 -y 0.0 -z 0.0 -R 0.0 -P 0.0 -Y 0.0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='gui',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='debug',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='public_sim',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='tuck_arm',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='recording',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='extra_gazebo_args',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='use_moveit_camera',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='use_dynamic_footprint',
            default_value='false'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='omni_base'
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
        launch_ros.actions.Node(
            package='ira_laser_tools',
            executable='laserscan_multi_merger',
            name='laserscan_multi_merger',
            parameters=[
                {
                    'destination_frame': 'virtual_base_laser_link'
                },
                {
                    'cloud_destination_topic': '/merged_cloud'
                },
                {
                    'scan_destination_topic': '/scan'
                },
                {
                    'laserscan_topics': '/scan_front_raw /scan_rear_raw'
                },
                {
                    'time_increment': '0.0'
                },
                {
                    'scan_time': '0.0'
                },
                {
                    'range_min': '0.05'
                },
                {
                    'range_max': '25.0'
                },
                {
                    'angle_min': '-3.1459'
                },
                {
                    'angle_max': '3.1459'
                },
                {
                    'angle_increment': '0.005769'
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_gazebo'), 'launch/tiago_spawn.launch.py')
            ),
            launch_arguments={
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'end_effector': launch.substitutions.LaunchConfiguration('end_effector'),
                'ft_sensor': launch.substitutions.LaunchConfiguration('ft_sensor'),
                'laser_model': launch.substitutions.LaunchConfiguration('laser_model'),
                'camera_model': launch.substitutions.LaunchConfiguration('camera_model'),
                'multi': launch.substitutions.LaunchConfiguration('multi'),
                'gzpose': launch.substitutions.LaunchConfiguration('gzpose'),
                'base_type': launch.substitutions.LaunchConfiguration('base_type')
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'tiago_gazebo'), 'launch/simulation_tiago_bringup.launch.py')
            ),
            launch_arguments={
                'arm': launch.substitutions.LaunchConfiguration('arm'),
                'end_effector': launch.substitutions.LaunchConfiguration('end_effector'),
                'ft_sensor': launch.substitutions.LaunchConfiguration('ft_sensor'),
                'laser_model': launch.substitutions.LaunchConfiguration('laser_model'),
                'camera_model': launch.substitutions.LaunchConfiguration('camera_model'),
                'public_sim': launch.substitutions.LaunchConfiguration('public_sim'),
                'use_moveit_camera': launch.substitutions.LaunchConfiguration('use_moveit_camera'),
                'use_dynamic_footprint': launch.substitutions.LaunchConfiguration('use_dynamic_footprint'),
                'base_type': launch.substitutions.LaunchConfiguration('base_type')
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
