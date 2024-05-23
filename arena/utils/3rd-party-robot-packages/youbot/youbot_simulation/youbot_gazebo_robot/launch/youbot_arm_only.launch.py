import os
import sys

import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='world',
            default_value='robocup_at_work_2012'
        ),
        launch.actions.DeclareLaunchArgument(
            name='init_pos_x',
            default_value='1.7'
        ),
        launch.actions.DeclareLaunchArgument(
            name='init_pos_y',
            default_value='0.0'
        ),
        launch.actions.DeclareLaunchArgument(
            name='init_pos_z',
            default_value='0.2'
        ),
        launch_ros.actions.Node(
            package='gazebo_ros',
            executable='spawn_model',
            name='spawn_robot',
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
                    'publish_frequency': '30.0'
                }
            ]
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'youbot_gazebo_worlds'), 'launch/$(arg world).launch.py')
            )
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'youbot_gazebo_control'), 'launch/joint_state_controller.launch.py')
            )
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'youbot_gazebo_control'), 'launch/arm_controller.launch.py')
            )
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'youbot_gazebo_control'), 'launch/gripper_controller.launch.py')
            )
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
