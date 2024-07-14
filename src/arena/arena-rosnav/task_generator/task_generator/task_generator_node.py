#! /usr/bin/env python3

import dataclasses
import os
import traceback
from typing import Dict, List

import rosros
import yaml
from rosros import RosPack
from task_generator.constants import Config, Constants
from task_generator.manager.entity_manager.entity_manager import EntityManager
from task_generator.manager.entity_manager.flatland_manager import FlatlandManager
from task_generator.manager.entity_manager.pedsim_manager import PedsimManager
from task_generator.manager.entity_manager.crowdsim_manager import CrowdsimManager

from task_generator.manager.world_manager import WorldManager
from task_generator.manager.obstacle_manager import ObstacleManager
from task_generator.manager.robot_manager import RobotManager
from task_generator.manager.utils import WorldMap
from task_generator.shared import (
    ModelWrapper,
    Namespace,
    Robot,
    gen_init_pos,
    rosparam_get
)
from task_generator.simulators.base_simulator import BaseSimulator
from task_generator.simulators.flatland_simulator import FlatlandSimulator  # noqa
from task_generator.simulators.gazebo_simulator import GazeboSimulator  # noqa
from task_generator.simulators.simulator_factory import SimulatorFactory
from task_generator.tasks import Task
from task_generator.tasks.task_factory import TaskFactory
from task_generator.utils import ModelLoader, Utils

from task_generator.manager.world_manager import WorldManager
from task_generator.manager.obstacle_manager import ObstacleManager

import map_distance_server.srv as map_distance_server_srvs
import std_msgs.msg as std_msgs
import std_srvs.srv as std_srvs


def create_default_robot_list(
    robot_model: ModelWrapper,
    name: str,
    inter_planner: str,
    local_planner: str,
    agent: str
) -> List[Robot]:
    return [
        Robot(
            model=robot_model,
            inter_planner=inter_planner,
            local_planner=local_planner,
            agent=agent,
            position=next(gen_init_pos),
            name=name,
            record_data_dir=rosparam_get(str, "record_data_dir", None),
            extra=dict(),
        )
    ]


def read_robot_setup_file(setup_file: str) -> List[Dict]:
    try:
        with open(
            os.path.join(
                RosPack().get_path("arena_bringup"),
                "configs",
                "robot_setup",
                setup_file,
            ),
            "r",
        ) as f:
            robots: List[Dict] = yaml.safe_load(f)["robots"]

        return robots

    except:
        traceback.print_exc()
        rosros.signal_shutdown("")
        raise Exception()


class TaskGenerator:
    """
    Task Generator Node
    Will initialize and reset all tasks. The task to use is read from the `/task_mode` param.
    """

    _entity_mode: Constants.EntityManager
    _auto_reset: bool

    _namespace: Namespace
    _env_wrapper: BaseSimulator
    _entity_manager: EntityManager
    _task: Task

    _pub_scenario_reset: rosros.Publisher
    _pub_scenario_finished: rosros.Publisher

    _start_time: float
    _number_of_resets: int

    def __init__(self, namespace: str = "/") -> None:
        self._namespace = Namespace(namespace)

        # Params

        self._entity_mode = Constants.EntityManager(rosparam_get(str, "entity_manager"))
        self._auto_reset = rosparam_get(bool, "~auto_reset", True)
        self._train_mode = rosparam_get(bool, "train_mode", False)

        # Publishers
        if not self._train_mode:
            self._pub_scenario_reset = rosros.create_publisher(
                std_msgs.Int16, "scenario_reset", qos_profile=1, latch=True
            )
            self._pub_scenario_finished = rosros.create_publisher(
                std_msgs.Empty, "scenario_finished", qos_profile=10
            )

            # Services
            rosros.create_service("reset_task", std_srvs.Empty, self._reset_task_srv_callback)

        # Vars
        self._env_wrapper = SimulatorFactory.instantiate(Utils.get_simulator())(
            namespace=self._namespace
        )

        # Loaders
        self._robot_loader = ModelLoader(
            os.path.join(RosPack().get_path("arena_simulation_setup"), "entities", "robots")
        )

        if not self._train_mode:
            self._start_time = rosros.get_time()
            self._task = self._get_predefined_task()
            rosros.set_param("/robot_names", self._task.robot_names)

            self._number_of_resets = 0

            self.srv_start_model_visualization = rosros.create_service_proxy(
                "start_model_visualization", std_srvs.Empty
            )
            self.srv_start_model_visualization(std_srvs.EmptyRequest())

            rosros.sleep(1)

            self.reset_task(first_map=True)

            rosros.sleep(1)

            try:
                rosros.set_param("task_generator_setup_finished", True)
                self.srv_setup_finished = rosros.create_service_proxy(
                    "task_generator_setup_finished", std_srvs.Empty
                )
                self.srv_setup_finished(std_srvs.EmptyRequest())
            except:
                pass

            # Timers
            rosros.create_timer(rosros.Duration(nsecs=int(0.5e9)), self._check_task_status)

        # SETUP

    def _get_predefined_task(self, **kwargs):
        """
        Gets the task based on the passed mode
        """
        if self._env_wrapper is None:
            self._env_wrapper = SimulatorFactory.instantiate(Utils.get_simulator())(
                self._namespace
            )

        rosros.wait_for_service("/distance_map")

        service_client_get_map = rosros.create_service_proxy(
            "/distance_map", map_distance_server_srvs.GetDistanceMap
        )

        map_response: map_distance_server_srvs.GetDistanceMapResponse = (
            service_client_get_map()
        )
        world_manager = WorldManager(
            world_map=WorldMap.from_distmap(distmap=map_response)
        )

        if self._entity_mode == Constants.EntityManager.PEDSIM:
            self._entity_manager = PedsimManager(
                namespace=self._namespace, simulator=self._env_wrapper
            )
        elif self._entity_mode == Constants.EntityManager.FLATLAND:
            self._entity_manager = FlatlandManager(
                namespace=self._namespace, simulator=self._env_wrapper
            )
        elif self._entity_mode == Constants.EntityManager.CROWDSIM:
            self._entity_manager = CrowdsimManager(
                namespace=self._namespace, simulator=self._env_wrapper
            )
        else:
            self._entity_manager = EntityManager(
                namespace=self._namespace, simulator=self._env_wrapper
            )

        obstacle_manager = ObstacleManager(
            namespace=self._namespace,
            world_manager=world_manager,
            simulator=self._env_wrapper,
            entity_manager=self._entity_manager,
        )

        obstacle_manager.spawn_world_obstacles(world_manager.world)

        robot_managers = self._create_robot_managers()

        # For every robot
        # - Create a unique namespace name
        # - Create a robot manager
        # - Launch the robot.launch file

        PARAM_TM_MODULES = "tm_modules"

        tm_modules_value = rosparam_get(str, PARAM_TM_MODULES, "")
        tm_modules = list(
            set(
                [
                    Constants.TaskMode.TM_Module(mod)
                    for mod in tm_modules_value.split(",")
                    if mod != ""
                ]
            )
        )

        tm_modules.append(Constants.TaskMode.TM_Module.CLEAR_FORBIDDEN_ZONES)
        tm_modules.append(Constants.TaskMode.TM_Module.RVIZ_UI)

        if rosparam_get(str, "map_file", "") == "dynamic_map":
            tm_modules.append(Constants.TaskMode.TM_Module.DYNAMIC_MAP)

        rosros.logdebug("utils calls task factory")
        task = TaskFactory.combine(
            modules=[Constants.TaskMode.TM_Module(module) for module in tm_modules]
        )(
            obstacle_manager=obstacle_manager,
            robot_managers=robot_managers,
            world_manager=world_manager,
            namespace=self._namespace,
            **kwargs,
        )

        return task

    def _create_robot_managers(self) -> List[RobotManager]:
        # Read robot setup file
        robot_setup_file: str = rosparam_get(str, "/robot_setup_file", "")

        robot_model: str = rosparam_get(str, "/model")


        if robot_setup_file == "":
            robots = create_default_robot_list(
                robot_model=self._robot_loader.bind(robot_model),
                inter_planner=rosparam_get(str, "/inter_planner", ""),
                local_planner=rosparam_get(str, "/local_planner", ""),
                agent=rosparam_get(str, "/agent_name", ""),
                name=f"{self._namespace[1:]}_{robot_model}"
                if self._train_mode
                else robot_model,
            )
        else:
            robots = [
                dataclasses.replace(
                    Robot.parse(
                        robot,
                        model=self._robot_loader.bind(robot["model"]),
                    ),
                    name=f'{robot["model"]}_{i}_{robot.get("amount", 1)-1}'
                )
                for robot in read_robot_setup_file(robot_setup_file)
                for i in range(robot.get("amount", 1))
            ]

        if Utils.get_arena_type() == Constants.ArenaType.TRAINING:
            return [
                RobotManager(
                    namespace=self._namespace,
                    entity_manager=self._entity_manager,
                    robot=robots[0],
                )
            ]

        robot_managers: List[RobotManager] = []

        for robot in robots:
            robot_managers.append(
                # RobotManager(os.path.join(namespace, name), simulator, robot)
                # old but working due to namespace issue with "/" prefix in robot name
                RobotManager(
                    namespace=self._namespace,
                    entity_manager=self._entity_manager,
                    robot=robot,
                )
            )

        return robot_managers

    # RUNTIME

    def reset_task(self, **kwargs):
        self._start_time = rosros.get_time()

        self._env_wrapper.before_reset_task()

        rosros.loginfo("resetting")

        is_end = self._task.reset(callback=lambda: False, **kwargs)

        self._env_wrapper.after_reset_task()

        self._pub_scenario_reset.publish(self._number_of_resets)
        self._number_of_resets += 1
        self._send_end_message_on_end()

        self._env_wrapper.after_reset_task()

        rosros.loginfo("=============")
        rosros.loginfo("Task Reset!")
        rosros.loginfo("=============")

    def _check_task_status(self, *args, **kwargs):
        if self._task.is_done:
            self.reset_task()

    def _reset_task_srv_callback(self, req: std_srvs.EmptyRequest):
        rosros.logdebug("Task Generator received task-reset request!")

        self.reset_task()

        return std_srvs.EmptyResponse()

    def _send_end_message_on_end(self):
        if self._number_of_resets < Config.General.DESIRED_EPISODES:
            return

        rosros.loginfo(f"Shutting down. All {int(Config.General.DESIRED_EPISODES)} tasks completed")

        rosros.signal_shutdown("Finished all episodes of the current scenario")


if __name__ == "__main__":
    rosros.init_node("task_generator")

    task_generator = TaskGenerator()

    rosros.spin()
