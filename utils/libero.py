import os
from typing import Dict, Generator, List, Tuple, Type, Union

import imageio
from libero.libero import benchmark
from libero.libero.benchmark import Benchmark, Task
from libero.libero.envs.env_wrapper import ControlEnv
import numpy
from sympy import true


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


def load_environment(task: Union[Task, None], 
                     bddl_file = "", 
                     has_render_onscreen = True, 
                     onscreen_camera_name = "frontview", 
                     has_camera_obs = True,
                     obs_camera_names: Union[str, List[str]] = ["agentview", "robot0_eye_in_hand", "sideview"],
                     camera_size = 224, 
                     ) -> ControlEnv:
    
    """ Loads a LIBERO.ControlEnv environment

        Loads a LIBERO from a Task or directly from a .bddl definition file 
        (in the end, the bddl file will be used, eventually from the Task passed)

        Parameters
        ----------
        task: Task or None
            The task to load in the environment or none to use the .bddl file
        bddl_file: str
            The path to the bddl file (set task to None to load the file)
        has_render_onscreen: bool
            If the render of the scene should appear on screen
        onscreen_camera_name: str
            Name of onscreen camera viewpoint. Avaialble viewpoint by default libero tasks are: 
            `frontview`, `birdview`, `agentview`, `sideview`
        has_camera_obs: bool
            If the evinroment should produce camera obserrvations
        obs_camera_names: str or list(str)
            The name of the camera (or a list of names) that will store camera observations - further research needed
            Camera name can surely be `agentview`, `robot0_eye_in_hand`, `sideview`. Can it be other?
        camera_size: int
            Resolution size of observation cameras

        Returns
        --------
        A `LIBERO.ControlEnv` object with the task loaded

    """

    # Checks the task is provided somehow
    assert task is not None or bddl_file != ""

    # Load bddl file
    if bddl_file == "":
        assert task is not None
        bddl_file = os.path.join(benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    os.path.exists(bddl_file)

    #print(f"Attempting to open {BDDL_FILE_PATH}")
    env = ControlEnv(
        bddl_file_name=bddl_file,
        has_renderer=has_render_onscreen,
        render_camera=onscreen_camera_name,
        has_offscreen_renderer= has_render_onscreen or has_camera_obs, # If we have any cameras we need offcreen render
        camera_names= obs_camera_names,
        camera_widths= camera_size,
        camera_heights= camera_size
    )
    return env


def save_video(frame_buffer: List[numpy.ndarray], path="", filename = "playback.mp4", fps = 30):
    """ Saves a list of images observations to video file

        Parameters
        -----------
        frame_buffer: List[numpy.ndarray]
            The buffer where obs are stored
        path: str
            The path where to save the video
        filename: str
            Default video filename
        fps: int
            The number of frames per second to save. Technically is should depend on inference time of the model,
            so default value of 30 will probably speed up the video a little
    """
    fullpath = os.path.join(path, filename)
    video_writer = imageio.get_writer(fullpath, fps=fps)
    for frame in frame_buffer:
        video_writer.append_data(frame)
    video_writer.close()