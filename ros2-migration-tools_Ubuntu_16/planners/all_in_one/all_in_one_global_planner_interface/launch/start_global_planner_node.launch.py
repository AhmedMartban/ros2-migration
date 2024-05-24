import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='ns',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='use_ns',
            default_value='true'
        ),
        launch.actions.DeclareLaunchArgument(
            name='config_path'
        ),
        launch.actions.DeclareLaunchArgument(
            name='node_name',
            default_value='global_planner_node'
        ),
        launch.actions.DeclareLaunchArgument(
            name='global_frame',
            default_value='map'
        ),
        launch.actions.DeclareLaunchArgument(
            name='robot_frame',
            default_value=launch.substitutions.LaunchConfiguration('ns')
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_footprint',
            default_value=launch.substitutions.LaunchConfiguration('ns')
        ),
        launch.actions.DeclareLaunchArgument(
            name='scan_frame',
            default_value=launch.substitutions.LaunchConfiguration('ns')
        ),
        launch_ros.actions.Node(
            package='all_in_one_global_planner_interface',
            executable='GlobalPlannerInterface',
            name=launch.substitutions.LaunchConfiguration('node_name'),
            output='screen',
            parameters=[
                launch.substitutions.LaunchConfiguration('config_path')
            ],
            condition=launch.conditions.IfCondition(
                "$(eval arg('use_ns') == true)")
        ),
        launch_ros.actions.Node(
            package='all_in_one_global_planner_interface',
            executable='GlobalPlannerInterface',
            name=launch.substitutions.LaunchConfiguration('node_name'),
            output='screen',
            parameters=[
                launch.substitutions.LaunchConfiguration('config_path')
            ],
            condition=launch.conditions.IfCondition(
                "$(eval arg('use_ns') == false)")
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
