import collections
from typing import List

import numpy

import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
CUSTOM_TASK = "/home/peraro/source/play-libero/libero-study/tasks/generated/PLATE_BOWL_SCENE_pick_up_the_central_black_bowl_and_place_it_on_the_plate.bddl"

# Envs
RENDER_ONSCREEN = True
CAMERA_NAME = "agentview"
MAX_STEPS = 600
CAMERA_SIZE = 224

# Video
VIDEO = False
VIDEO_PATH = "./videos/openpi/custom-tasks/"

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
    print(obs.keys())
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

    print()
    task_env.close()

    if VIDEO: utils.libero.save_video(frame_buffer, path=VIDEO_PATH, filename=f"{task_env.problem_name}.mp4")
        
if __name__ == "__main__":
    main()