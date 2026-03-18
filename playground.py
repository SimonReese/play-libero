import collections
import os
from tkinter.tix import MAX
from typing import Dict, Generator, List, Tuple, Type

from git import HEAD, Tree
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

def main():

    # Load all tasks
    for bench, task in utils.libero.load_tasks(BENCHMARK_SUITE):
        print(f"Loading task: {task.name}")
        print(f"Instruction: {task.language}")
        # Load task environment
        task_env = utils.libero.load_environment(task, onscreen_camera_name="agentview", has_render_onscreen=RENDER_ONSCREEN)
        # Reset env
        task_env.seed(0)
        obs: collections.OrderedDict = task_env.reset()

        # Let env run 10 steps to allow objects to fall
        for step in range(10):
            if RENDER_ONSCREEN: task_env.env.render()
            obs, reward, done, info = task_env.step(DUMMY_ACTION)

        # Step over the environment
        frame_buffer: List[numpy.ndarray] = []
        for step in range(MAX_STEPS):
            if RENDER_ONSCREEN: task_env.env.render()
            if step % 100 == 0: print("*", end="", flush=True)
            
            # Execute dummy action (randomly open-close gripper)
            DUMMY_ACTION[-1] = numpy.random.uniform(-1, 1)
            obs, reward, done, info = task_env.step(DUMMY_ACTION)
            
            # Apparently, we need to flip images from obs :-) :-| :-< >:-<
            agentview_image = obs["agentview_image"][::-1]
            wrist_image = obs["robot0_eye_in_hand_image"][::-1]
            sideview_image = obs["sideview_image"][::-1]

            if VIDEO: frame_buffer.append(agentview_image)

        task_env.close()

        if VIDEO: utils.libero.save_video(frame_buffer, path=VIDEO_PATH, filename=f"{task.name}.mp4")
    
if __name__ == "__main__":
    main()