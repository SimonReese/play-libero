import collections
import os
from typing import Dict, Generator, List, Tuple, Type

import imageio
from libero.libero import benchmark
from libero.libero.benchmark import Benchmark, Task
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
import numpy
import openpi_client
import openpi_client.websocket_client_policy
import robosuite
import robosuite.utils
import robosuite.utils.transform_utils
from openpi_client import image_tools

from playground import RENDER_ONSCREEN
import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING

# Envs
RENDER_ONSCREEN = True
CAMERA_NAME = "agentview"
CAMERA_SIZE = 224
MAX_STEPS = 600

# Video
VIDEO = False
VIDEO_PATH = "./videos/openpi/"

# Connection
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000

# Action
DUMMY_ACTION = [0.] * 7


def main():
    # Open websocket
    remote_model = openpi_client.websocket_client_policy.WebsocketClientPolicy(TITAN_IP, TITAN_PORT)
    
    # Load all tasks
    for bench, task in utils.libero.load_tasks(BENCHMARK_SUITE):
        print(f"Loading task: {task.name}")
        print(f"Instruction: {task.language}")
        # Load task environment
        task_env = utils.libero.load_environment(task, has_render_onscreen=RENDER_ONSCREEN, onscreen_camera_name=CAMERA_NAME)
        # Reset env
        task_env.seed(0)
        obs: collections.OrderedDict = task_env.reset()

        # Wait 10 steps for objects to fall
        done = False
        for step in range(10):
            # Perform dummy action
            obs, reward, done, info = task_env.step(DUMMY_ACTION)

        # step over the environment
        frame_buffer: List[numpy.ndarray] = []
        for step in range(MAX_STEPS):
            if done:
                break

            # Prepare images
            front_img = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = numpy.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            front_img = image_tools.convert_to_uint8(front_img)
            wrist_img = image_tools.convert_to_uint8(wrist_img)
            # Pack obs
            observations = {
                "observation/image": front_img,
                "observation/wrist_image": wrist_img,
                "observation/state": numpy.concatenate(
                    (obs["robot0_eef_pos"], 
                    robosuite.utils.transform_utils.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"])
                ),
                "prompt": task.language
            }
            # Get action
            action_chunk = remote_model.infer(observations)["actions"]

            # Perform each action in action chunk
            action: numpy.ndarray
            for action in action_chunk:
                if RENDER_ONSCREEN: task_env.env.render()
                if VIDEO: frame_buffer.append(front_img)
                if step % 10 == 0: print("*", end="", flush=True)
                obs, reward, done, info = task_env.step(action.tolist())
                # Increase step
                step +=1

        task_env.close()

        if VIDEO: utils.libero.save_video(frame_buffer, filename=f"{task.name}.mp4")
        
if __name__ == "__main__":
    main()