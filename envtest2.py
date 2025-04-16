import os
import numpy as np
import gym
import time
import dexterous_gym
import gym_emg
import pathlib

# Calibrated data for DB1, DB2 and DB5 is DB9: https://ninapro.hevs.ch/instructions/DB9.html

# Test environments
name = 'gym_emg/SingleHand-v0'
name = 'gym_emg/TwoHands-v0'
print(name)
datapath = f"{pathlib.Path('~').expanduser()}/Desktop/COMP579/data"
env = gym.make(name, datapath=datapath, subject=2, exercise=2) # This will also take care of registering the environment


d = False
while not d:
    obs = env.reset() 
    env.render()
    r_sum = 0.0
    d = False
    while not d:
        action = env.action_space.sample()
        #action = np.zeros(len(env.action_space.sample()))
        obs, r, d, info = env.step(action)
        r_sum += r
        #time.sleep(0.07)
        env.render()
    print(r_sum)
env.close()