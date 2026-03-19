import collections
from typing import List

import numpy

import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
""" libero_spatial
        for spatial understaning, same task different objects
    libero_object
        different layout with same objects
    libero_goal
        different goal, same layout and objs
    libero_90
        training set of libero_100
    libero_10
        eval set of libero_100
    libero_100
"""

# Envs
RENDER_ONSCREEN = False
CAMERA_NAME = "agentview"
CAMERA_SIZE = 512
MAX_STEPS = 100

# Video
VIDEO = True
VIDEO_PATH = f"./videos/playground/{BENCHMARK_SUITE}/"

# Action
DUMMY_ACTION = [0.] * 7 

def main():

    for BENCHMARK_SUITE in ["libero_object", "libero_goal", "libero_90", "libero_10", "libero_100"]:
        VIDEO_PATH = f"./videos/playground/{BENCHMARK_SUITE}/"
        print(f"Starting benchmark {BENCHMARK_SUITE}")
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

            print() # End line
            task_env.close()
            # Save video
            if VIDEO: utils.libero.save_video(frame_buffer, path=VIDEO_PATH, filename=f"{task.name}.mp4")
        # end task
    # end benchmark
    
if __name__ == "__main__":
    main()