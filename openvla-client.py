import collections
from typing import List

import numpy
from openpi_client import image_tools

import requests
import json_numpy

import utils
json_numpy.patch()

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING

# Envs
RENDER_ONSCREEN = True
CAMERA_NAME = "agentview"
CAMERA_SIZE = 224
MAX_STEPS = 600

# Video
VIDEO = False
VIDEO_PATH = "./videos/openvla/"

# Connection
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000

# Action
DUMMY_ACTION = [0.] * 7 


def main():
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
            # Perform dummy actionR: task_env.
            obs, reward, done, info = task_env.step(DUMMY_ACTION)

        # step over the environment
        frame_buffer: List[numpy.ndarray] = []
        for step in range(MAX_STEPS):
            if done:
                break

            # Prepare images
            front_img = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1]) # Flip
            front_img = image_tools.convert_to_uint8(front_img) # Convert to uint8
            
            # Pack obs
            observations = {
                "image": front_img,
                "instruction": task.language
            }
            
            # Get action
            action: numpy.ndarray = requests.post(
                url=f"http://{TITAN_IP}:{TITAN_PORT}/act",
                json=observations
            ).json()

            # Trasform action to libero env
            action = utils.model.normalize_gripper_action(action)
            action = utils.model.invert_gripper_action(action)

            if RENDER_ONSCREEN: task_env.env.render()
            if VIDEO: frame_buffer.append(front_img)
            if step % 10 == 0: print("*", end="", flush=True)
            
            # Perform action
            obs, reward, done, info = task_env.step(action.tolist())
     
        task_env.close()

        if VIDEO: utils.libero.save_video(frame_buffer, filename=f"{task.name}.mp4")
    
if __name__ == "__main__":
    main()