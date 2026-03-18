import collections
import os
from typing import Dict, Generator, List, Tuple, Type

import imageio
import numpy
from libero.libero import benchmark
from libero.libero.benchmark import Benchmark, Task
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
from openpi_client import image_tools

import requests
import json_numpy
json_numpy.patch()

BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
HEADLESS = True
VIDEO = True
VIDEO_PATH = "./videos/openvla/"
MAX_STEPS = 600
CAMERA_NAME = "agentview"
CAMERA_SIZE = 224
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000
DUMMY_ACTION = [0.] * 7 

def normalize_gripper_action(old_action: numpy.ndarray, binarize=True):
    action = numpy.copy(old_action)
    """ From https://github.com/openvla/openvla/blob/main/experiments/robot/robot_utils.py#L75

    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = numpy.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """ From https://github.com/openvla/openvla/blob/main/experiments/robot/robot_utils.py#L95

    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

def load_task(benchmark_suite: str, task_id: int) -> Tuple[Benchmark, Task]:
    ''' Load a spedfic task from a specific benchmark of LIBERO

        Parameters
        -------------- 
        benchmark_suite: str
            The benchmark/task suite to load. Available benchmarks are:

                `libero_spatial`
                    for spatial understaning, same task different objects
                `libero_object`
                    different layout with same objects
                `libero_goal`
                    different goal, same layout and objs
                `libero_90` 
                    training set of `libero_100`
                `libero_10`
                    eval set of `libero_100`
                `libero_100`
                    different objects, layouts and scenes
        task_id: int
            The id of the task to load

        Returns
        -------
        A Tuple[Benchmark, Task] or a generated list of Tuple[Benchmark, Task] for each possible task.

        A Task object is simply a tuple of:
        - `name`: task name (_ separated)
        - `language`: tasbenchmark.k lang instruction
        - `problem`: libero tasks are under `libero` problem
        - `problem_folder`: the specific `libero_benckmark` folder name
        - `bddl_file`: the full .bbdl file name (no path)
        - `init_states_file`: the .init file name for the task
    '''
    
    benchmark_dict: Dict[str, Type[Benchmark]] = benchmark.get_benchmark_dict()
    ''' The `benchmark_dict` (mapped with the variable `benchmark.BENCHMARK_MAPPING`) 
        containst the available Libero Benchmarks Suites, that are:
        - `libero_spatial` : for spatial understaning, same task different objects
        - `libero_object`: different layaot with same objects
        - `libero_goal` : different goal, same layout and objs
        - `libero_90` : training set of `libero_100`
        - `libero_10` : eval set of `libero_100`
        - `libero_100` : different objects, layouts and scenes 
    '''

    task_suite: Benchmark = benchmark_dict[benchmark_suite]()
    '''`task_suite` will be an instance of one of the available Benchmarks classes
        Each benchmark has 10 tasks, except `benchmark.LIBERO_90` which has 90 tasks
    '''

    # Print information about the selected benchmark
    print(f"Loaded: {task_suite}, {task_suite.get_num_tasks()} tasks available")
    print(f"Available tasks: {task_suite.get_task_names()}")

    # Load a specific task
    task: Task = task_suite.get_task(task_id)
    return task_suite, task

def load_tasks(benchmark_suite: str) -> Generator[Tuple[Benchmark, Task], None, None]:
    # Return the first task
    task_suite, task = load_task(benchmark_suite, 0)
    yield task_suite, task
    # Generate and return all the remaining tasks
    for task_id in range(1, task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        yield task_suite, task

def load_tasks_from(benchmark_suite: str, from_id: int) -> Generator[Tuple[Benchmark, Task], None, None]:
     # Return the first task
    task_suite, task = load_task(benchmark_suite, from_id)
    yield task_suite, task
    # Generate and return all the remaining tasks
    for task_id in range(from_id + 1, task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        yield task_suite, task

def load_environment(task: Task) -> ControlEnv:

    # Load bddl file
    BDDL_FILE_PATH = os.path.join(benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    #print(f"Attempting to open {BDDL_FILE_PATH}")

    env_args = {
        "bddl_file_name": BDDL_FILE_PATH,
        "camera_heights": CAMERA_SIZE,
        "camera_widths": CAMERA_SIZE
    }

    if HEADLESS:
        env = OffScreenRenderEnv(**env_args)
    else:
        env = ControlEnv(BDDL_FILE_PATH, 
                         has_renderer=True, # On screen view
                         has_offscreen_renderer=True, # If camera or headless, an offscreen is required
                         render_camera=CAMERA_NAME,
                         camera_heights=CAMERA_SIZE,
                         camera_widths=CAMERA_SIZE
                         )
    return env

def save_video(frame_buffer: List[numpy.ndarray], filename = "playback.mp4", fps = 30):
    video_writer = imageio.get_writer(f"{VIDEO_PATH}{filename}", fps=fps)
    for frame in frame_buffer:
        video_writer.append_data(frame)
    video_writer.close()


def main():
    # Load all tasks
    for bench, task in load_tasks_from(BENCHMARK_SUITE, 4):
        print(f"Loading task: {task.name}")
        print(f"Instruction: {task.language}")
        # Load task environment
        task_env = load_environment(task)
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
                "instruction": task.language
            }
            
            # Get action
            action: numpy.ndarray = requests.post(
                url=f"http://{TITAN_IP}:{TITAN_PORT}/act",
                json=observations
            ).json()
            # Trasform action to libero env
            action = normalize_gripper_action(action)
            action = invert_gripper_action(action)

            if not HEADLESS: task_env.env.render()
            if VIDEO: frame_buffer.append(front_img)
            if step % 10 == 0: print("*", end="", flush=True)
            
            # Perform action
            obs, reward, done, info = task_env.step(action.tolist())
     
        print(f"Observations: {obs.keys()}")
        task_env.close()

        if VIDEO: save_video(frame_buffer, filename=f"{task.name}.mp4")
    
if __name__ == "__main__":
    main()