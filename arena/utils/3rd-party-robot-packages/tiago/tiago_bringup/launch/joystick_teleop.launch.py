import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='True'
        ),
        launch.actions.DeclareLaunchArgument(
            name='end_effector',
            default_value='pal-hey5'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ft_sensor',
            default_value='schunk-ft'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type_suffix',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='cmd_vel',
            default_value='input_joy/cmd_vel'
        ),
        launch.actions.DeclareLaunchArgument(
            name='file_suffix',
            default_value="$(eval (('no-arm' if not arm else (end_effector if end_effector else 'no-ee') + ('_' +  ft_sensor if ft_sensor else ''))))"
        ),
        launch.actions.DeclareLaunchArgument(
            name='teleop_config',
            default_value=launch.substitutions.LaunchConfiguration(
                'file_suffix')
        ),
        launch.actions.DeclareLaunchArgument(
            name='teleop_config_move',
            default_value=launch.substitutions.LaunchConfiguration(
                'base_type_suffix')
        ),
        launch_ros.actions.Node(
            package='joy_teleop',
            executable='incrementer_server',
            name='incrementer',
            condition=launch.conditions.IfCondition(
                "$(eval end_effector in ['pal-gripper', 'schunk-wsg', 'robotiq-2f-85', 'robotiq-2f-140'])")
        ),
        launch_ros.actions.Node(
            package='joy_teleop',
            executable='joy_teleop.py',
            name='joy_teleop'
        ),
        launch_ros.actions.Node(
            package='joy',
            executable='joy_node',
            name='joystick',
            parameters=[
                {
                    'autorepeat_rate': '10'
                }
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
