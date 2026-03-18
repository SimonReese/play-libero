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

BENCHMARK_SUITE = "libero_spatial" # one from benchmark.BENCHMARK_MAPPING
HEADLESS = False
VIDEO = True
VIDEO_PATH = "./videos/openpi/"
MAX_STEPS = 600
CAMERA_NAME = "agentview"
CAMERA_SIZE = 224
TITAN_IP = "titan2.dei.unipd.it"
TITAN_PORT = 8000
DUMMY_ACTION = [0.] * 7

# Open websocket
remote_model = openpi_client.websocket_client_policy.WebsocketClientPolicy(TITAN_IP, TITAN_PORT)


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
    for bench, task in load_tasks(BENCHMARK_SUITE):
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
                if not HEADLESS: task_env.env.render()
                if VIDEO: frame_buffer.append(front_img)
                if step % 10 == 0: print("*", end="", flush=True)
                obs, reward, done, info = task_env.step(action.tolist())
                # Increase step
                step +=1


        print(f"Observations: {obs.keys()}")
        task_env.close()

        if VIDEO: save_video(frame_buffer, filename=f"{task.name}.mp4")
        
if __name__ == "__main__":
    main()