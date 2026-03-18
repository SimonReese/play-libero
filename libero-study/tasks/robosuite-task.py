import xml
import xml.etree
import xml.etree.ElementTree

import robosuite
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.environments.base import REGISTERED_ENVS
import mujoco

world = MujocoWorldBase()
panda = Panda()
gripper = gripper_factory("PandaGripper")
panda.add_gripper(gripper)

panda.set_base_xpos((0, 0, 0))
world.merge(panda)


arena = TableArena()
arena.set_origin((0, 0, 0))
arena.set_camera("front", (0, 0, 0), (1, 0, 0, 0))
world.merge(arena)

ball: xml.etree.ElementTree.Element = BallObject("ball", size=[0.06], rgba=[0.9, 0, 0, 1]).get_obj()
ball.set('pos', "1.0 0.0 1.0")
world.worldbody.append(ball)

model = world.get_model()

data = mujoco.MjData(model)
while data.time < 10:
    mujoco.mj_step(model, data) # type: ignore


import numpy as np
import robosuite as suite

# create environment instance
print(REGISTERED_ENVS)
env = suite.make(
    env_name="NutAssembly", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(*env.action_spec[0].shape) * 0.1
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display