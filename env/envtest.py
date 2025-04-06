import mujoco_py
import os
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

envs = [
    'EggHandOver-v0',
    'BlockHandOver-v0',
    'PenHandOver-v0',
    'EggHandOverSparse-v0',
    'BlockHandOverSparse-v0',
    'PenHandOverSparse-v0',
    'EggCatchUnderarm-v0',
    'BlockCatchUnderarm-v0',
    'PenCatchUnderarm-v0',
    'EggCatchUnderarmHard-v0',
    
    """
    'BlockCatchUnderarmHard-v0',
    'PenCatchUnderarmHard-v0',
    'EggCatchUnderarmSparse-v0',
    'BlockCatchUnderarmSparse-v0',
    'PenCatchUnderarmSparse-v0',
    'EggCatchUnderarmHardSparse-v0',
    'BlockCatchUnderarmHardSparse-v0',
    'PenCatchUnderarmHardSparse-v0',
    'TwoEggCatchUnderArm-v0',
    'TwoBlockCatchUnderArm-v0',
    'TwoPenCatchUnderArm-v0',
    'TwoEggCatchUnderArmSparse-v0',
    'TwoBlockCatchUnderArmSparse-v0',
    'TwoPenCatchUnderArmSparse-v0',
    'EggCatchOverarm-v0',
    'BlockCatchOverarm-v0',
    'PenCatchOverarm-v0',
    'EggCatchOverarmSparse-v0',
    'BlockCatchOverarmSparse-v0',
    'PenCatchOverarmSparse-v0',
    'PenSpin-v0'
    """
]

for name in envs:

    env = gym.make(name)
    print(name)

    for j in range(1):
        obs = env.reset()
        env.render()
        r_sum = 0.0
        for i in range(50):
            obs, r, d, info = env.step(env.action_space.sample())
            r_sum += r
            time.sleep(0.07)
            env.render()
        print(r_sum)
    env.close()