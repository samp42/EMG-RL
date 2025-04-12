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


# Test load data
import scipy.io
file = "/home/jacobyroy/Desktop/COMP579/s3/S3_E2_A1.mat"
mat = scipy.io.loadmat(file)


# TODO: normalize data to range accepted by sim
for element in mat:
    print(element)
print(mat)

print(mat["glove"].shape)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 8-bit proportional to joint angle according to this: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186132
# BUT index 10 (ID 11 of the glove, sensor between index and middle) is outside of range
#plt.plot(mat["glove"][:,10])
#plt.show()

"""
Mapping for glove is found here: https://static1.squarespace.com/static/559c381ee4b0ff7423b6b6a4/t/5602fbd2e4b07ebf58d47ecd/1443036114091/CyberGloveII_Brochure.pdf
https://zenodo.org/records/1000116

Note that the glove only has flexion movement, not lateral movements.

Mapping for sim should be this (we have 20 instead of 23 though): https://robotics.farama.org/envs/adroit_hand/adroit_pen/

"""

# NOTE: glove index is MATLAB index, need -1
sim2glove = {
    1:22,  # Wrist flex
    2:None, # Wrist flex (lateral)
    3:5, # Index metacarpal
    4:6, # Index proximal
    5:None,
    6:9,  # Middle metacarpal
    7:10, # Middle proximal
    8:None,
    9:13,  # Ring metacarpal
    10:14,  # Ring proximal
    11:None,
    12:None,
    13:17, # Pinky metacarpal
    14:18, # Pinky proximal
    15:None, # Thumb axe 1
    16:None, # Thumb axe 2
    17:None, # Thumb lateral
    18:2, # Thumb metacarpal
    19:3, # Thumb proximal
}

# CyberGlove II
sim2glove = {
    1:21,  # Wrist flex
    2:22, # Wrist flex (lateral)
    3:5, # Index metacarpal
    4:7, # Index proximal
    5:None,
    6:8,  # Middle metacarpal
    7:9, # Middle proximal
    8:None,
    9:10,  # Ring metacarpal
    10:12,  # Ring proximal
    11:None,
    12:None,
    13:14, # Pinky metacarpal
    14:16, # Pinky proximal
    15:None, # Thumb axe 1
    16:None, # Thumb axe 2
    17:None, # Thumb lateral
    18:1, # Thumb metacarpal
    19:3, # Thumb proximal
}


# NOTE: assume metacarpal and proximal have 0 to 100 degrees (-1 to 1) flexion
# Sim has only 0 to 90?
sim2glove = {
    "WF": { # Wrist flex
        "id": {1:21},
        "map": [[-1,1],[0,128]]
    },
    "WFl": { # Wrist flex (lateral)
        "id": {2:22},
        "map": [[-1,1],[0,128]]
    },

    "IM": { # Index metacarpal 
        "id": {3:7}, # OK
        "map": [[-1,1],[0,100]] 
    },
    "IP": { # Index proximal 
        "id": {4:5}, # OK
        "map": [[-1,1],[0,100]] 
    },

    "Ma": { # Middle abduction
        "id": {5:None},
        "map": [[-1,1],[0,128]]
    },
    "MM": { # Middle metacarpal
        "id": {6:8}, # OK
        "map": [[-1,1],[0,128]]
    },
    "MP": { # Middle proximal
        "id": {7:9}, # OK
        "map": [[-1,1],[0,128]]
    },

    "Ra": { # Ring abduction
        "id": {8:None},
        "map": [[-1,1],[0,128]]
    },
    "RM": { # Ring metacarpal
        "id": {9:10},
        "map": [[-1,1],[0,128]]
    },
    "RP": { # Ring proximal
        "id": {10:12},
        "map": [[-1,1],[0,128]]
    },

    "Pa1": { # Pinky abduction 1
        "id": {11:None},
        "map": [[-1,1],[0,128]]
    },
    "Pa2": { # Pinky abduction 2
        "id": {12:None},
        "map": [[-1,1],[0,128]]
    },
    "PM": { # Pinky metacarpal
        "id": {13:14},
        "map": [[-1,1],[0,128]]
    },
    "PP": { # Pinky proximal
        "id": {14:16},
        "map": [[-1,1],[0,128]]
    },


    "Ta1": { # Thumb abduction 1
        "id": {15:None},
        "map": [[-1,1],[0,128]]
    },
    "Ta2": { # Thumb abduction 2
        "id": {16:None},
        "map": [[-1,1],[0,128]]
    },
    "Ta3": { # Thumb abduction 3
        "id": {17:None},
        "map": [[-1,1],[0,128]]
    },
    "TM": { # Thumb metacarpal
        "id": {18:1},
        "map": [[-1,1],[0,128]]
    },
    "TP": { # Thumb proximal
        "id": {19:3},
        "map": [[-1,1],[0,128]]
    },
}


glove2sim = {}
for joint in sim2glove:
    if list(sim2glove[joint]["id"].values())[0] is not None:
        # Fix index with -1
        glove2sim[joint] = {
            "id": {list(sim2glove[joint]["id"].values())[0]-1 : list(sim2glove[joint]["id"].keys())[0]},
            "map": [sim2glove[joint]["map"][1], sim2glove[joint]["map"][0]]
        }
        
print(glove2sim)

glove2sim_ID = {}
glove2sim_MAP = list(np.zeros(22))
for joint in glove2sim:
    glove2sim_ID[list(glove2sim[joint]["id"].keys())[0]] = list(glove2sim[joint]["id"].values())[0]
    glove2sim_MAP[list(glove2sim[joint]["id"].keys())[0]]  = glove2sim[joint]["map"]

print(glove2sim_ID)
print(glove2sim_MAP)

# Keep only relevant data
#joints = mat["glove"][:,list(glove2sim_ID.keys())]
#maps = [glove2sim_MAP[i] for i in list(glove2sim_ID.keys())]
plt.plot(mat["glove"][:, [0,1,2,3]])
plt.show()

exit()

# Map values from glove to sim (account for 8-bit values and range of motion): [-128, 127] to [-1, 1]
for i in range(joints.shape[1]): # Apply linear mapping (gain + offset)
    m = interp1d(maps[i][0], maps[i][1], fill_value="extrapolate")
    joints[:,i] = m(joints[:,i])
#plt.plot(joints[:,2])
#plt.show()

# Clip from -1 to 1 (because range of motion of sim is smaller than what is recorded)
joints = np.clip(joints, -1, 1)
plt.plot(joints[:,[4,5]])
plt.show()

# Test environments
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

        """
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
        """

        
        """
        # Test single joint
        joint = 2
        for pos in range(5):
            action[joint] = pos*0.2 - 1
            obs, r, d, info = env.step(action)
            r_sum += r
            time.sleep(0.07)
            env.render()

        for pos in range(5):
            action[joint] = pos*0.2
            obs, r, d, info = env.step(action)
            r_sum += r
            time.sleep(0.07)
            env.render()
        """

        # INDEX FLEXION IS (+), EXTENSION IS (-) (joints ID 5 is index in glove)
        # Test the data
        freq = mat["frequency"][0][0] 
        action = np.zeros(len(env.action_space.sample()))
        # Too many samples to render in real-time, skip samples to go faster
        skip = 0
        print(list(glove2sim_ID.keys()))
        print(list(glove2sim_ID.values()))
        for sample in joints:
            skip += 1
            if skip % 25 == 0:
                #action[list(glove2sim_ID.values())] = sample

                action[3] = sample[2] # Index metacarpal
                action[4] = sample[3] # Index proximal

                action[6] = sample[4] # Middle metacarpal
                action[7] = sample[5] # Middle proximal
                print(sample[2])
                obs, r, d, info = env.step(action)
                r_sum += r
                time.sleep(1/freq)
                env.render()
    print(r_sum)
env.close()