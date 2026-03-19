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
CUSTOM_TASK = "/home/peraro/source/play-libero/libero-study/tasks/PLATE_SCENE_pick_up_cookie_box_and_place_it_on_the_top_right_plate.bddl"

# Envs
RENDER_ONSCREEN = True
MAX_STEPS = 600
CAMERA_NAME = "agentview"
CAMERA_SIZE = 224

# Video
VIDEO = True
VIDEO_PATH = "./videos/openvla/custom-tasks/"

# Connection
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000

# Action
DUMMY_ACTION = [0.] * 7


def main():

    # Load task environment
    task_env = utils.libero.load_environment(None, CUSTOM_TASK)
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
        front_img = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1]) # Flip
        front_img = image_tools.convert_to_uint8(front_img) # Convert to uint8
        # Pack obs
        observations = {
            "image": front_img,
            "instruction": task_env.language_instruction
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

    if VIDEO: utils.libero.save_video(frame_buffer, path=VIDEO_PATH, filename=f"{task_env.problem_name}.mp4")
    
if __name__ == "__main__":
    main()