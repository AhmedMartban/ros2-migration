import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='end_effector'
        ),
        launch.actions.DeclareLaunchArgument(
            name='local_joint_control',
            default_value='direct_effort_control'
        ),
        launch.actions.DeclareLaunchArgument(
            name='controller_pkg_name',
            default_value='tiago_controller_configuration'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ns',
            default_value='gravity_compensation'
        ),
        launch.actions.DeclareLaunchArgument(
            name='wrist_model',
            default_value='wrist-2010'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ee_suffix',
            default_value="$(eval (end_effector if end_effector else 'no-ee'))"
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
                    'pal_local_joint_control'), 'launch/no_control.launch.py')
            ),
            launch_arguments={
                'controller_pkg_name': launch.substitutions.LaunchConfiguration('controller_pkg_name'),
                'ns': launch.substitutions.LaunchConfiguration('ns'),
                'joint_name': 'torso_lift_joint'
            }.items()
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
