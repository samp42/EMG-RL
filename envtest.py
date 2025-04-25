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

# This will also take care of registering the environment
# subsampling will subsample the dataset (keep at 1 if want all samples)
# n_substeps will affect simulation accuracy and stability by performing more simulated steps in-between steps
# subject and exercise determine the datasets to be used
env = gym.make(name, datapath=datapath, n_substeps=5, subsampling=1, subject=2, exercise=3) # with rendering


# NOTE: recommended to use only exercise 2, as exercise 3 has objects which provide force feedback (EMG may not match as expected without force feedback)
subjects = {
    1:[2,3],
    2:[2,3],
    3:[2,3],
    4:[2,3],
    5:[2,3],
    6:[2,3],
    7:[2,3],
    8:[2,3],
    9:[2,3],
    10:[2,3],
}

for subject in subjects:
    env = gym.make(name, datapath=datapath, n_substeps=5, subsampling=1, subject=subject, exercise=2) 
    obs = env.reset() 
    env.render()

    # Episodes
    env.set_mode("train") # set env in training mode
    episodes = np.array(range(env.get_num_trials()))
    np.random.shuffle(episodes) # select each trial randomly
    for ep in episodes:
        env.draw(ep) # draw current episode in trials

        r_sum = 0.0
        d = False
        #while not d:
        while not d: 
            action = env.action_space.sample()
            #action = np.zeros(len(env.action_space.sample()))
            obs, r, d, info = env.step(action)
            r_sum += r
            print(r)
            #time.sleep(0.07)
            env.render() # TODO: remove for faster simulation
        print(r_sum)
        env.close()