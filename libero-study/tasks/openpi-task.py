import collections
import os
from typing import List

from exceptiongroup import catch
import numpy
import openpi_client
import openpi_client.websocket_client_policy
import robosuite
import robosuite.utils
import robosuite.utils.transform_utils
from openpi_client import image_tools

import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
CUSTOM_TASK = "/home/peraro/source/play-libero/libero-study/tasks/generated/PLATE_BOWL_SCENE_pick_up_the_central_black_bowl_and_place_it_on_the_plate.bddl"
CUSTOM_TASK_FOLDER = "/home/peraro/source/play-libero/libero-study/tasks/generated"

# Envs
RENDER_ONSCREEN = False
CAMERA_NAME = "agentview"
MAX_STEPS = 600
CAMERA_SIZE = 224

# Video
VIDEO = True
VIDEO_PATH = "./videos/openpi/custom-tasks/PLATE_BOWL_SCENE/sideview"

# Connection
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000

# Action
DUMMY_ACTION = [0.] * 7

def main():
    # Open websocket
    remote_model = openpi_client.websocket_client_policy.WebsocketClientPolicy(TITAN_IP, TITAN_PORT)

    for FILE in os.listdir(CUSTOM_TASK_FOLDER):
        if "PLATE_BOWL_SCENE" not in FILE:
            continue
        CUSTOM_TASK= os.path.join(CUSTOM_TASK_FOLDER, FILE)
        task_name = FILE.strip(".bddl")

        # Load task environment
        task_env = utils.libero.load_environment(None, CUSTOM_TASK)
        print(f"Opening task: {task_name}")
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
        step = 0
        while step < MAX_STEPS:
            if done:
                break

            # Prepare images
            front_img = numpy.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = numpy.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            side_img = numpy.ascontiguousarray(obs["sideview_image"][::-1, ::-1])
            front_img = image_tools.convert_to_uint8(front_img)
            wrist_img = image_tools.convert_to_uint8(wrist_img)
            side_img = image_tools.convert_to_uint8(side_img)
            # Pack obs
            observations = {
                "observation/image": side_img,
                "observation/wrist_image": wrist_img,
                "observation/state": numpy.concatenate(
                    (obs["robot0_eef_pos"], 
                    robosuite.utils.transform_utils.quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"])
                ),
                "prompt":  "with respect to the front camera, pick the left plate" #task_env.language_instruction
            }
            # Get action
            action_chunk = remote_model.infer(observations)["actions"]

            # Perform each action in action chunk
            action: numpy.ndarray
            for action in action_chunk:
                if done: break
                if RENDER_ONSCREEN: task_env.env.render()
                if VIDEO: frame_buffer.append(side_img)
                if step % 100 == 0: print("*", end="", flush=True)
                try:
                    obs, reward, done, info = task_env.step(action.tolist())
                except ValueError as err:
                    print(f"Error: {err} at {step}, {done}")
                    done = False
                    step = MAX_STEPS
                    break
                # Increase step
                step +=1

        print()
        task_env.close()
        print(f"Result: {done}")

        if VIDEO: utils.libero.save_video(frame_buffer, path=VIDEO_PATH, filename=f"{task_name}.mp4")

        # Store result
        with open(f"{VIDEO_PATH}/results.csv", "a") as result_file:
            result_file.write(f"openpi, {task_name}, {done}\n")
        result_file.close()

        
if __name__ == "__main__":
    main()