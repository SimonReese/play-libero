import collections
import os
from typing import Dict, Generator, List, Tuple, Type

from attr import dataclass
import imageio
from libero.libero import benchmark
from libero.libero.benchmark import Benchmark, Task
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
import numpy

import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING

# Envs
RENDER_ONSCREEN = True
CAMERA_NAME = "agentview"
CAMERA_SIZE = 512
MAX_STEPS = 100

# Video
VIDEO = True
VIDEO_PATH = f"./videos/playground/{BENCHMARK_SUITE}/"

# Action
DUMMY_ACTION = [0.] * 7 


@dataclass
class TaskEntry:
    task_name: str
    task_instruction: str
    

@dataclass
class BenchmarkEntry:
    tasks: Dict[int, TaskEntry]

    def to_json_dict(self):
        return{task_id:task for task_id, task in self.tasks.items()}


def main():
    # Load every benchmark
    for benchmark_suite in benchmark.BENCHMARK_MAPPING.keys():
        
        # Load all tasks
        for bench, task in utils.libero.load_tasks(benchmark_suite):
            print(f"Loading task: {task.name}")
            print(f"Instruction: {task.language}")
            # Load task environment
            task_env = utils.libero.load_environment(task)
            # Reset env
            task_env.seed(0)
            obs: collections.OrderedDict = task_env.reset()

            # step over the environment
            frame_buffer: List[numpy.ndarray] = []
            for step in range(100):
                if RENDER_ONSCREEN: task_env.env.render()
                if step % 100 == 0: print("*", end="", flush=True)
                
                # Execute dummy action (randomly open-close gripper)
                DUMMY_ACTION[-1] = numpy.random.uniform(-1, 1)
                obs, reward, done, info = task_env.step(DUMMY_ACTION)
                
                # Apparently, we need to flip images from obs :-) :-| :-< >:-<
                agentview_image = obs["agentview_image"][::-1]
                wrist_image = obs["robot0_eye_in_hand_image"][::-1]

            task_env.close()

    # TODO: construct a dataset made of {benchmark name, task name, task video, lang instruction}

    
if __name__ == "__main__":
    main()