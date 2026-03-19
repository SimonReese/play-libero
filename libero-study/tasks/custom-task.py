
import re
from tkinter import OFF
from typing import ClassVar, Dict, List, Tuple, Type

from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.base_object import OBJECTS_DICT
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import MU_DICT, SCENE_DICT, InitialSceneTemplates, register_mu
from libero.libero.utils.object_utils import get_affordance_regions
from libero.libero.utils.task_generation_utils import TASK_INFO, TaskInfoTuple, generate_bddl_from_task_info, register_task_info
import numpy


GENERATED_BDDL_PATH = "/home/peraro/source/play-libero/libero-study/tasks/generated/"

"""print("Objects:", OBJECTS_DICT.keys())
print("Predicates:", VALIDATE_PREDICATE_FN_DICT.keys())"""

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
class DrawerPlateCookieScene(InitialSceneTemplates):
    """
        A scene with a cookiebox in a drawer and 9 plates

        Plates scheme wrt agentview camera:
            1 2 3 <--goal plate
            4 5 6
            7 8 9
    """

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
    
    @classmethod
    def get_scene_name(cls) -> str:
        """ Returns the class name in LIBERO format. Useful to have when registering the class"""
        return "_".join(re.sub(r"([A-Z])", r" \1", cls.__name__).split()).lower()
    
    @classmethod
    def scene_instructions(cls) -> List[str]:
        """ Returns available task instructions for this scene"""
        return [
            "Pick up cookie box and place it on the top right plate"
        ]
    
    @classmethod
    def goal_states(cls) -> List[List[Tuple[str, str, str]]]:
        # Define a set of goal statements for every instruction
        all_goals=[]
        # Instruction 1
        goal_1: List[Tuple[str, str, str]] = [("On", "cookies_1", "plate_3")]
        
        all_goals.append(goal_1)
        return all_goals

@register_mu(scene_type="kitchen")
class PlateCookieScene(InitialSceneTemplates):
    """
        A scene with three cookiebox and a central plate

        Scheme wrt agentview camera:
            
            c1 p c2
               c3 
    """

    def __init__(self):
        fixtures = {
            "kitchen_table": 1,
        }

        objs = {
            "plate": 1,
            "cookies": 3
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixtures,
            object_num_info=objs
        )
        
    def define_regions(self):

        plate_region = self.get_region_dict(
            region_centroid_xy=[0.0, 0],
            region_name="plate_region",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        OFFSET = 0.15
        cookies_region_1 = self.get_region_dict(
            region_centroid_xy=[0.0, -OFFSET],
            region_name="cookies_region_1",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        cookies_region_2 = self.get_region_dict(
            region_centroid_xy=[0, OFFSET],
            region_name="cookies_region_2",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        cookies_region_3 = self.get_region_dict(
            region_centroid_xy=[OFFSET, 0],
            region_name="cookies_region_3",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        self.regions.update(plate_region)
        self.regions.update(cookies_region_1)
        self.regions.update(cookies_region_2)
        self.regions.update(cookies_region_3)

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
            ("On", "cookies_1", "kitchen_table_cookies_region_1"),
            ("On", "cookies_2", "kitchen_table_cookies_region_2"),
            ("On", "cookies_3", "kitchen_table_cookies_region_3"),
            ("On", "plate_1", "kitchen_table_plate_region"),
        ]
        return states
    
    @classmethod
    def get_scene_name(cls) -> str:
        """ Returns the class name in LIBERO format. Useful to have when registering the class"""
        return "_".join(re.sub(r"([A-Z])", r" \1", cls.__name__).split()).lower()
    
    @classmethod
    def scene_instructions(cls) -> List[str]:
        """ Returns available task instructions for this scene"""
        return [
            "Pick up all cookie boxes and place them on the plate"
        ]
    
    @classmethod
    def goal_states(cls) -> List[List[Tuple[str, str, str]]]:
        # Define a set of goal statements for every instruction
        all_goals=[]
        # Instruction 1
        goal_1: List[Tuple[str, str, str]] = [
            ("On", "cookies_1", "plate_1"),
            ("On", "cookies_2", "plate_1"),
            ("On", "cookies_3", "plate_1"),
        ]
        
        all_goals.append(goal_1)
        return all_goals
    
@register_mu(scene_type="kitchen")
class PlateBowlScene(InitialSceneTemplates):
    """
        A scene with three cookiebox and a central plate

        Scheme wrt agentview camera:
            
            b1 p b2
               b3 
    """

    def __init__(self):
        fixtures = {
            "kitchen_table": 1,
        }

        objs = {
            "plate": 1,
            "akita_black_bowl": 3
        }

        super().__init__(
            workspace_name="kitchen_table",
            fixture_num_info=fixtures,
            object_num_info=objs
        )
        
    def define_regions(self):
        
        OFFSET = 0.2

        plate_region = self.get_region_dict(
            region_centroid_xy=[-OFFSET, 0],
            region_name="plate_region",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )
        
        bowl_region_1 = self.get_region_dict(
            region_centroid_xy=[-OFFSET, -OFFSET],
            region_name="bowl_region_1",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        bowl_region_2 = self.get_region_dict(
            region_centroid_xy=[-OFFSET, OFFSET],
            region_name="bowl_region_2",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        bowl_region_3 = self.get_region_dict(
            region_centroid_xy=[0, 0],
            region_name="bowl_region_3",
            target_name=self.workspace_name,
            region_half_len=0.0001
        )

        self.regions.update(plate_region)
        self.regions.update(bowl_region_1)
        self.regions.update(bowl_region_2)
        self.regions.update(bowl_region_3)

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
            ("On", "akita_black_bowl_1", "kitchen_table_bowl_region_1"),
            ("On", "akita_black_bowl_2", "kitchen_table_bowl_region_2"),
            ("On", "akita_black_bowl_3", "kitchen_table_bowl_region_3"),
            ("On", "plate_1", "kitchen_table_plate_region"),
        ]
        return states
    
    @classmethod
    def get_scene_name(cls) -> str:
        """ Returns the class name in LIBERO format. Useful to have when registering the class"""
        return "_".join(re.sub(r"([A-Z])", r" \1", cls.__name__).split()).lower()
    
    @classmethod
    def scene_instructions(cls) -> List[str]:
        """ Returns available task instructions for this scene"""
        return [
            "Pick up all black bowls and place them on the plate",  # Goal 1

            "Pick up the central black bowl and place it on the plate", # Goal 2
            "Pick up the furthest black bowl from robot and place it on the plate", # Goal 2
            "With respect to the frontal camera pick up the black bowl which is in front of the plate and place it on the plate", # Goal 2

            "Pick up the left black bowl with respect to the frontal camera and place it on the plate", # Goal 3
            "Pick up the right black bowl with respect to the robot and place it on the plate", # Goal 3

            "With respect to the frontal camera pick up the black bowl on the left of the plate and place it on the plate", # Goal 3
            "With respect to the robot pick up the black bowl on the right of the plate and place it on the plate", # Goal 3

            "Pick up the right black bowl with respect to the frontal camera and place it on the plate", # Goal 4
            "Pick up the left black bowl with respect to the robot and place it on the plate", # Goal 4

            "With respect to the frontal camera pick up the black bowl on the right of the plate and place it on the plate", # Goal 4
            "With respect to the robot pick up the black bowl on the left of the plate and place it on the plate", # Goal 4

        ]
    
    @classmethod
    def goal_states(cls) -> List[List[Tuple[str, str, str]]]:
        # Define a set of goal statements for every instruction
        all_goals=[]
        # Instruction 1
        goal_1: List[Tuple[str, str, str]] = [
            ("On", "akita_black_bowl_1", "plate_1"),
            ("On", "akita_black_bowl_2", "plate_1"),
            ("On", "akita_black_bowl_3", "plate_1"),
        ]

        # Instruction 2, 3, 4
        goal_2: List[Tuple[str, str, str]] = [
            ("On", "akita_black_bowl_3", "plate_1"),
        ]

        # Instruction 5, 8
        goal_3: List[Tuple[str, str, str]] = [
            ("On", "akita_black_bowl_1", "plate_1"),
        ]

        # Instruction 6
        goal_4: List[Tuple[str, str, str]] = [
            ("On", "akita_black_bowl_2", "plate_1"),
        ]
        
        all_goals.append(goal_1)
        all_goals.append(goal_2)
        all_goals.append(goal_3)
        all_goals.append(goal_4)
        return all_goals
    
    @classmethod
    def register_all(cls):
        # Goal 1
        register_task_info(
            PlateBowlScene.scene_instructions()[0], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[0] # type:ignore
        )

        # Goal 2 -------------------------
        register_task_info(
            PlateBowlScene.scene_instructions()[1], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[1] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[2], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[1] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[3], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[1] # type:ignore
        )

        # Goal 3 -------------------------
        register_task_info(
            PlateBowlScene.scene_instructions()[4], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[2] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[5], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[2] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[6], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[2] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[7], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[2] # type:ignore
        )

        # Goal 4 -------------------------
        register_task_info(
            PlateBowlScene.scene_instructions()[8], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[3] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[9], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[3] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[10], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[3] # type:ignore
        )

        register_task_info(
            PlateBowlScene.scene_instructions()[11], # type: ignore
            PlateBowlScene.get_scene_name(), # type: ignore
            objects_of_interest=["akita_black_bowl_1", "akita_black_bowl_2", "akita_black_bowl_3"],
            goal_states=PlateBowlScene.goal_states()[3] # type:ignore
        )


        

def main():
 
    # Register DrawerPlateCookieScene
    register_task_info(
        DrawerPlateCookieScene.scene_instructions()[0], # type: ignore
        DrawerPlateCookieScene.get_scene_name(), # type: ignore
        objects_of_interest=["cookies_1"],
        goal_states=DrawerPlateCookieScene.goal_states()[0] # type:ignore
    )

    register_task_info(
        PlateCookieScene.scene_instructions()[0], # type: ignore
        PlateCookieScene.get_scene_name(), # type: ignore
        objects_of_interest=["cookies_1"],
        goal_states=PlateCookieScene.goal_states()[0] # type:ignore
    )

    PlateBowlScene.register_all() # type: ignore

    bddl_file_names, failures = generate_bddl_from_task_info(folder=GENERATED_BDDL_PATH)
    print(f"Generated {bddl_file_names}")
    print(f"Failures {failures}")


if __name__ == "__main__":
    main()


"""print(f"MU_DICT: {MU_DICT}")
print(f"SCENE_DICT: {SCENE_DICT}")
print(f"TASK_INFO: {TASK_INFO}")
print(get_affordance_regions(OBJECTS_DICT, True))"""