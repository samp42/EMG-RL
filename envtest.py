import mujoco_py
import os
import numpy as np
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]

import gym
import time
import dexterous_gym
import gym_emg

envs = [
    'PenSpin-v0',
]



name = 'gym_emg/SingleHand-v0'
name = 'gym_emg/TwoHands-v0'
print(name)
env = gym.make(name) # This will also take care of registering the environment

for j in range(1):
    obs = env.reset()
    env.render()
    r_sum = 0.0
    d = False
    while not d:
        action = env.action_space.sample()
        action = np.zeros(len(env.action_space.sample()))
        obs, r, d, info = env.step(action)
        r_sum += r
        time.sleep(0.07)
        env.render()

        # Test each joint from -1 to 1
        for joint in range(len(env.action_space.sample())):
            for pos in range(5):
                action[joint] = pos*0.1 - 1
                obs, r, d, info = env.step(action)
                r_sum += r
                time.sleep(0.07)
                env.render()

            for pos in range(5):
                action[joint] = 1 - pos*0.1
                obs, r, d, info = env.step(action)
                r_sum += r
                time.sleep(0.07)
                env.render()
    print(r_sum)
env.close()