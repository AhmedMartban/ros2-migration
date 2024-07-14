#! /usr/bin/env python3

import rosros
from task_generator.task_generator_node import TaskGenerator

if __name__ == "__main__":
    rosros.init_node("task_generator")

    task_generator = TaskGenerator()

    rosros.spin()
