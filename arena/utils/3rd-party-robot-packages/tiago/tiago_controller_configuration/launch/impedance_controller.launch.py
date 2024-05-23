import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='end_effector',
            default_value='pal-hey5'
        ),
        launch.actions.DeclareLaunchArgument(
            name='wrist_model',
            default_value='wrist-2017'
        ),
        launch.actions.DeclareLaunchArgument(
            name='ee_suffix',
            default_value="$(eval (end_effector if end_effector else 'no-ee'))"
        ),
        launch_ros.actions.Node(
            package='controller_manager',
            executable='spawner',
            name='impedance_controller_spawner',
            output='screen',
            condition=launch.conditions.IfCondition(
                "$(eval wrist_model == 'wrist-2017')")
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
