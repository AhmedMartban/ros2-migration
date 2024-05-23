import os
import sys

import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='multi',
            default_value=''
        ),
        launch.actions.DeclareLaunchArgument(
            name='arm',
            default_value='True'
        ),
        launch.actions.DeclareLaunchArgument(
            name='wrist_model',
            default_value='wrist-2017'
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
            name='laser_model',
            default_value='sick-571'
        ),
        launch.actions.DeclareLaunchArgument(
            name='camera_model',
            default_value='orbbec-astra'
        ),
        launch.actions.DeclareLaunchArgument(
            name='no_safety_eps',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='robot_description',
            default_value='robot_description'
        ),
        launch.actions.DeclareLaunchArgument(
            name='base_type',
            default_value='pmb2'
        ),
        launch.actions.DeclareLaunchArgument(
            name='description_calibration_dir',
            default_value=os.environ.get(
                'pal_eye_hand_calibration_tiago_calibration_path')
        ),
        launch.actions.DeclareLaunchArgument(
            name='extrinsic_calibration_dir',
            default_value=os.environ.get(
                'pal_camera_calibration_extrinsics_calibration_path')
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
