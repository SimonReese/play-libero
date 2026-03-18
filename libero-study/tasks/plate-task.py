
from typing import ClassVar, Dict, List, Tuple, Type

from libero.libero.envs import objects
from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.base_object import OBJECTS_DICT
from libero.libero.envs.bddl_base_domain import TASK_MAPPING
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import MU_DICT, SCENE_DICT, InitialSceneTemplates, register_mu
from libero.libero.utils.object_utils import get_affordance_regions
from libero.libero.utils.task_generation_utils import TASK_INFO, TaskInfoTuple, generate_bddl_from_task_info, register_task_info
import numpy


print("Objects:", OBJECTS_DICT.keys())
print("Predicates:", VALIDATE_PREDICATE_FN_DICT.keys())

"""
    This file creates a scene with 9 plates, a wooden cabinet and a cookie box.
    Scene reference frames:
        WRT frontview camera, the reference frame of the table used as reference fixture is:

        +------------------- > y
        |
        |
        |
        |
        |
        |
        |
        |
        ⌄
        x

"""

# Related variables
TASK_INFO: Dict[str, List[TaskInfoTuple]]
"""Stores information about generated tasks.
    A dictionary structured as:
    - `scene_name`: [TaskInfoTuple, ...]

    TaskInfoTuple is storing (scene_name language objects_of_interest goal_states)
"""

MU_DICT: Dict[str, Type[InitialSceneTemplates]]
"""Stores registered classes names and corresponding types.
    A dictionary structured as:
    - `scene_name`: Type
"""

SCENE_DICT: Dict[str, Type[InitialSceneTemplates]]
"""Stores registered classes names and corresponding types.
    A dictionary structured as:
    - `scene_name`: Type
"""


@register_mu(scene_type="kitchen")
class PlateScene(InitialSceneTemplates):

    def __init__(self):
        fixtures = {
            "kitchen_table": 1,
            "wooden_cabinet": 1,
        }

        objs = {
            "plate": 9,
            "cookies": 1
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixtures,
            object_num_info=objs
        )
        
    def define_regions(self):
        
        cabinet_region = self.get_region_dict(
            region_centroid_xy=[0.0, -0.4],
            region_name="cabinet_region",
            target_name=self.workspace_name,
            region_half_len=0.0001,
            yaw_rotation=(numpy.pi, numpy.pi)
        )
        """
            Plates scheme:
                1 2 3
                4 5 6
                7 8 9
        """
        # -----------------------------------
        PLATE_OFF = 0.15
        plate_region_1 = self.get_region_dict(
            region_centroid_xy=[-PLATE_OFF, -PLATE_OFF],
            region_name="plate_region_1",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_2 = self.get_region_dict(
            region_centroid_xy=[-PLATE_OFF, 0],
            region_name="plate_region_2",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_3 = self.get_region_dict(
            region_centroid_xy=[-PLATE_OFF, PLATE_OFF],
            region_name="plate_region_3",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        # -----------------------------------
        plate_region_4 = self.get_region_dict(
            region_centroid_xy=[0.0, -PLATE_OFF],
            region_name="plate_region_4",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_5 = self.get_region_dict(
            region_centroid_xy=[0.0, 0],
            region_name="plate_region_5",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_6 = self.get_region_dict(
            region_centroid_xy=[0, PLATE_OFF],
            region_name="plate_region_6",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        # ----------------------------------
        plate_region_7 = self.get_region_dict(
            region_centroid_xy=[PLATE_OFF, -PLATE_OFF],
            region_name="plate_region_7",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_8 = self.get_region_dict(
            region_centroid_xy=[PLATE_OFF, 0],
            region_name="plate_region_8",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        plate_region_9 = self.get_region_dict(
            region_centroid_xy=[PLATE_OFF, PLATE_OFF],
            region_name="plate_region_9",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        # --------------------------------
 
        cookies_region = self.get_region_dict(
            region_centroid_xy=[0.3, 0.3],
            region_name="cookie_region",
            target_name=self.workspace_name,
            region_half_len=0.001
        )

        self.regions.update(plate_region_1)
        self.regions.update(plate_region_2)
        self.regions.update(plate_region_3)
        self.regions.update(plate_region_4)
        self.regions.update(plate_region_5)
        self.regions.update(plate_region_6)
        self.regions.update(plate_region_7)
        self.regions.update(plate_region_8)
        self.regions.update(plate_region_9)

        self.regions.update(cabinet_region)
        self.regions.update(cookies_region)

        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        states: List[Tuple[str, ...]]
        """A list of valid initialization statement.
            A statement is made of:
            - predicate one from VALIDATE_PREDICATE_FN_DICT
            - arg(s) one or more arguments for the predicate

            An arg can be an object or a target region in case 
        """
        states = [
            ("On", "plate_1", "kitchen_table_plate_region_1"),
            ("On", "plate_2", "kitchen_table_plate_region_2"),
            ("On", "plate_3", "kitchen_table_plate_region_3"),
            ("On", "plate_4", "kitchen_table_plate_region_4"),
            ("On", "plate_5", "kitchen_table_plate_region_5"),
            ("On", "plate_6", "kitchen_table_plate_region_6"),
            ("On", "plate_7", "kitchen_table_plate_region_7"),
            ("On", "plate_8", "kitchen_table_plate_region_8"),
            ("On", "plate_9", "kitchen_table_plate_region_9"),
            ("On", "wooden_cabinet_1", "kitchen_table_cabinet_region"),
            ("In", "cookies_1", "wooden_cabinet_1_top_region"),
            ("Open", "wooden_cabinet_1_top_region")
        ]
        return states

scene_name = "plate_scene"
scene_language = "Pick up cookie box and place it on the top right plate"
register_task_info(
    scene_language,
    scene_name,
    objects_of_interest=["cookies_1"],
    goal_states=[
        ("On", "cookies_1", "plate_3")
    ]
)

print(f"MU_DICT: {MU_DICT}")
print(f"SCENE_DICT: {SCENE_DICT}")
print(f"TASK_INFO: {TASK_INFO}")

TEMP_BDDL_PATH = "/home/peraro/source/play-libero/libero-study/tasks"
bddl_file_names, failures = generate_bddl_from_task_info(folder=TEMP_BDDL_PATH)
print(failures)
print(bddl_file_names)


print(get_affordance_regions(OBJECTS_DICT, True))