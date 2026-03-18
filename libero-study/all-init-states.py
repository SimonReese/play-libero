import collections
from typing import List

import numpy

import utils

# Libero
BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
TASK_ID = 0

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
    bench, task = utils.libero.load_task(BENCHMARK_SUITE, TASK_ID)
    print(f"Loading task: {task.name}")
    print(f"Instruction: {task.language}")
    # Load task environment
    task_env = utils.libero.load_environment(task, onscreen_camera_name="agentview", has_render_onscreen=RENDER_ONSCREEN)
    # Reset env
    task_env.seed(0)
    obs: collections.OrderedDict = task_env.reset()

    # Get all init states
    init_states = bench.get_task_init_states(TASK_ID)
    for index, init in enumerate(init_states):
        print(f"State {index}/{len(init_states)}")
        task_env.set_init_state(init)

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

            if VIDEO: frame_buffer.append(agentview_image)

        if VIDEO: utils.libero.save_video(frame_buffer, filename=f"{task.name}.mp4")
    task_env.close()
    
if __name__ == "__main__":
    main()