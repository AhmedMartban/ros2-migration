import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='local_joint_control_type',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='ns',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='local_joint_control',
            default_value='direct_position_control'
        ),
        launch.actions.DeclareLaunchArgument(
            name='controller_pkg_name',
            default_value='tiago_controller_configuration'
        ),
        launch.actions.DeclareLaunchArgument(
            name='control_torso',
            default_value='true'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_1_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_2_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_3_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_4_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_5_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_6_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'arm_7_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'head_1_joint'
            }.items()
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'pal_local_joint_control'), 'launch/$(arg local_joint_control).launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'head_2_joint'
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
