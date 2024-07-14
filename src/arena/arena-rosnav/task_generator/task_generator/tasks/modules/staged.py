import dataclasses
import os
from typing import Any, Dict, List, NamedTuple, Optional

import dynamic_reconfigure.client
import rospkg
import rosros
import std_msgs.msg as std_msgs
import yaml
from filelock import FileLock
from map_generator.constants import MAP_GENERATOR_NS
from task_generator.constants import Constants
from task_generator.shared import Namespace, rosparam_get
from task_generator.tasks.modules import TM_Module
from task_generator.tasks.task_factory import TaskFactory

class Stage(NamedTuple):
    static: int
    interactive: int
    dynamic: int
    goal_radius: Optional[float]
    dynamic_map: Optional["DynamicMapStage"]

    def serialize(self) -> Dict:
        return self._asdict()

class DynamicMapStage(NamedTuple):
    algorithm: str
    algorithm_config: Dict[str, Any]

    def serialize(self) -> Dict:
        return self._asdict()

StageIndex = int
Stages = Dict[StageIndex, Stage]

@dataclasses.dataclass
class Config:
    stages: Stages
    starting_index: StageIndex

@TaskFactory.register_module(Constants.TaskMode.TM_Module.STAGED)
class Mod_Staged(TM_Module):
    __config: Config
    __target_stage: StageIndex
    __current_stage: StageIndex

    __training_config_path: Optional[Namespace]
    __debug_mode: bool
    __config_lock: FileLock

    PARAM_CURR_STAGE = "/curr_stage"
    PARAM_LAST_STAGE_REACHED = "/last_state_reached"
    PARAM_GOAL_RADIUS = "/goal_radius"
    PARAM_DEBUG_MODE = "debug_mode"

    PARAM_CURRICULUM = "STAGED_curriculum"
    PARAM_INDEX = "STAGED_index"

    PARAM_CONFIGURATION_NAME = lambda obs_type, param: f"RANDOM_{obs_type}_{param}"

    TOPIC_PREVIOUS_STAGE = "previous_stage"
    TOPIC_NEXT_STAGE = "next_stage"

    CONFIG_PATH: Namespace
    CURRICULUM_PATH: Namespace

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.CONFIG_PATH = Namespace(
            os.path.join(
                rospkg.RosPack().get_path("arena_bringup"), "configs", "training"
            )
        )

        self.CURRICULUM_PATH = self.CONFIG_PATH("training_curriculums")

        self.__debug_mode = rosparam_get(bool, "debug_mode", False)

        self.__training_config_path = (
            rosparam_get(str, "training_config_path", None)
            if not self.__debug_mode
            else None
        )

        def cb_next(*args, **kwargs):
            self.stage_index += 1

        rosros.Subscriber(
            os.path.join(
                self._TASK.namespace,
                self.TOPIC_NEXT_STAGE,
            ),
            std_msgs.Bool,
            cb_next,
        )

        def cb_previous(*args, **kwargs):
            self.stage_index -= 1

        rosros.Subscriber(
            os.path.join(
                self._TASK.namespace,
                self.TOPIC_PREVIOUS_STAGE,
            ),
            std_msgs.Bool,
            cb_previous,
        )

        if self.__training_config_path is not None:
            assert os.path.isfile(
                self.__training_config_path
            ), f"Found no 'training_config.yaml' at {self.__training_config_path}"

            self.__config_lock = FileLock(f"{self.__training_config_path}.lock")

        self.__current_stage = -1

        self._dmre_client = dynamic_reconfigure.client.Client(
            name=self.NODE_CONFIGURATION, config_callback=self.reconfigure
        )

    def before_reset(self):
        if self.__current_stage != self.__target_stage:
            self.__current_stage = self.__target_stage
            rosros.loginfo(
                f"[{self._TASK.namespace}] Loading stage {self.__current_stage}"
            )

            if "sim_1" in rosros.get_name() or self.__debug_mode:
                goal_radius = self.stage.goal_radius
                if goal_radius is None:
                    goal_radius = rosparam_get(float, self.PARAM_GOAL_RADIUS, 0.3)
                rosros.set_param(self.PARAM_GOAL_RADIUS, goal_radius)

                if self.stage.dynamic_map.algorithm is not None:
                    rosros.set_param(
                        MAP_GENERATOR_NS("algorithm"), self.stage.dynamic_map.algorithm
                    )
                if self.stage.dynamic_map.algorithm_config is not None:
                    rosros.set_param(
                        MAP_GENERATOR_NS("algorithm_config"),
                        self.stage.dynamic_map.algorithm_config,
                    )

                obs_config = {}
                for obs_type in ["static", "dynamic", "interactive"]:
                    obs_config.update(
                        {
                            Mod_Staged.PARAM_CONFIGURATION_NAME(
                                obs_type, param
                            ): getattr(self.stage, obs_type)
                            for param in ["min", "max"]
                        }
                    )

                self._dmre_client.update_configuration(obs_config)

            if self.__training_config_path is not None:
                pass

    def reconfigure(self, config):
        try:
            curriculum_file = str(self.CURRICULUM_PATH(config[self.PARAM_CURRICULUM]))
        except Exception as e:
            rosros.logwarn(e)
            curriculum_file = "default.yaml"

        assert os.path.isfile(curriculum_file), f"{curriculum_file} is not a file"

        with open(curriculum_file) as f:
            stages = {
                i: Stage(
                    static=stage.get("static", 0),
                    interactive=stage.get("interactive", 0),
                    dynamic=stage.get("dynamic", 0),
                    goal_radius=stage.get("goal_radius", None),
                    dynamic_map=DynamicMapStage(
                        algorithm=stage["map_generator"].get("algorithm"),
                        algorithm_config=stage["map_generator"].get("algorithm_config"),
                    ),
                )
                for i, stage in enumerate(yaml.load(f, Loader=yaml.FullLoader))
            }

        try:
            starting_index = config[self.PARAM_INDEX]
        except Exception as e:
            rosros.logwarn(e)
            starting_index = 0

        self.__config = Config(stages=stages, starting_index=starting_index)
        self.stage_index = starting_index

    @property
    def IS_EVAL_SIM(self) -> bool:
        return "eval_sim" in self._TASK.namespace

    @property
    def MIN_STAGE(self) -> StageIndex:
        return 0

    @property
    def MAX_STAGE(self) -> StageIndex:
        return len(self.__config.stages) - 1

    @property
    def stage_index(self) -> StageIndex:
        return self.__current_stage

    @stage_index.setter
    def stage_index(self, val: StageIndex):
        val = val if val is not None else self.MIN_STAGE

        if val < self.MIN_STAGE or val > self.MAX_STAGE:
            rosros.loginfo(
                f"({self._TASK.namespace}) INFO: Tried to set stage {val} but was out of bounds [{self.MIN_STAGE}, {self.MAX_STAGE}]"
            )
            val = max(self.MIN_STAGE, min(self.MAX_STAGE, val))

        self.__target_stage = val

        if self.IS_EVAL_SIM and self.__current_stage != self.__target_stage:
            rosros.set_param(self.PARAM_CURR_STAGE, self.__target_stage)
            rosros.set_param(
                self.PARAM_LAST_STAGE_REACHED,
                self.__target_stage == self.MAX_STAGE,
            )
            os.system(
                f"rosrun dynamic_reconfigure dynparam set /task_generator_server {self.PARAM_INDEX} {self.__target_stage}"
            )

    @property
    def stage(self) -> Stage:
        return self.__config.stages[self.stage_index]
