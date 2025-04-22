import scipy.io
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# CyberGlove II mapping: https://www.researchgate.net/figure/Cyberglove-II-device-and-the-placement-of-the-22-sensors_fig1_338483027
sim2glove_old = {
    "WF": { # Wrist flex
        "id": {1:21},
        "glovemap": [0,100], # Raw data to Angles (min/max of raw data used)
        "simmap": [0, 90] # Angle to sim range (-1 to 1)
    },
    "WFl": { # Wrist flex (lateral)
        "id": {2:22},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },

    "IM": { # Index metacarpal
        "id": {3:5}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "IP": { # Index proximal
        "id": {4:6}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },

    "Ma": { # Middle abduction
        "id": {5:None},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "MM": { # Middle metacarpal
        "id": {6:8}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "MP": { # Middle proximal
        "id": {7:9}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },

    "Ra": { # Ring abduction
        "id": {8:None},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "RM": { # Ring metacarpal
        "id": {9:12}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "RP": { # Ring proximal
        "id": {10:13}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },

    "Pa1": { # Pinky abduction 1
        "id": {11:20},
        "glovemap": [0,20],
        "simmap": [0, 40]
    },
    "Pa2": { # Pinky abduction 2
        "id": {12:None},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "PM": { # Pinky metacarpal
        "id": {13:16}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "PP": { # Pinky proximal
        "id": {14:17}, # OK
        "glovemap": [0,100],
        "simmap": [0, 90]
    },


    # 1: general hand flexion? more like metacarpal to/from palm
    #

    "Ta": { # Thumb abduction (proximal)
        "id": {17:None},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "TM1": { # Thumb metacarpal 1 (lateral to palm, but with rotation??)
        #"id": {15:1},
        "id": {15:None},
        "glovemap": [0,100],
        "simmap": [0, 90] # OK
    },
    "TM2": { # Thumb metacarpal 2 (to/from palm)
        #"id": {16:4},
        "id": {16:None},
        "glovemap": [0,100],
        "simmap": [0, 45] # OK
    },
    "TP": { # Thumb proximal
        "id": {18:2}, # OK
        #"id": {18:None},
        "glovemap": [90,0],
        "simmap": [0, 90] # OK
    },
    "Tb": { # Thumb ptit boutte
        "id": {19:3}, # OK
        "glovemap": [100,0],
        "simmap": [0, 90]
    },
}




sim2glove = {
    "WF": { # Wrist flex
        "id": {1:21},
        "simmap": [0, 90] # Calibrated angle to sim range (-1 to 1)
    },
    "WFl": { # Wrist flex (lateral)
        "id": {2:22},
        "simmap": [0, 90]
    },

    "IM": { # Index metacarpal
        "id": {3:5}, # OK
        "simmap": [0, 90] # OK
    },
    "IP": { # Index proximal
        "id": {4:7}, # OK
        "simmap": [0, 140]
    },

    "Ma": { # Middle abduction
        #"id": {5:6},
        "id": {5:None},
        "simmap": [-20, 20]
    },
    "MM": { # Middle metacarpal
        "id": {6:8},
        "simmap": [0, 90]
    },
    "MP": { # Middle proximal
        "id": {7:9},
        #"simmap": [0, 90] # Not working??
        "simmap": [0, 60]
    },

    "Ra": { # Ring abduction
        #"id": {8:11},
        "id": {8:None},
        "simmap": [-20, 20]
    },
    "RM": { # Ring metacarpal
        "id": {9:10}, # OK
        "simmap": [0, 110] # unsure about range
    },
    "RP": { # Ring proximal
        "id": {10:12},
        "simmap": [0, 160]
    },

    "Pa1": { # Pinky abduction 1
        #"id": {11:20},
        "id": {11:None},
        "simmap": [0, 40]
    },
    "Pa2": { # Pinky abduction 2
        #"id": {12:15},
        "id": {12:None},
        "simmap": [0, 90]
    },
    "PM": { # Pinky metacarpal
        "id": {13:14}, # OK
        "simmap": [0, 150]
    },
    "PP": { # Pinky proximal
        "id": {14:16}, # OK
        "simmap": [0, 90]
    },

    "Ta": { # Thumb abduction (proximal)
        "id": {17:None},
        "simmap": [0, 90]
    },
    "TM1": { # Thumb metacarpal 1 (lateral to palm, but with rotation??)
        "id": {15:2},
        "simmap": [0,40]
    },
    "TM2": { # Thumb metacarpal 2 (to/from palm)
        "id": {16:1},
        "simmap": [-40,45]
    },
    "TP": { # Thumb proximal
        "id": {18:3},
        "simmap": [-25, 20]
    },
    "Tb": { # Thumb ptit boutte
        "id": {19:4},
        "simmap": [65, -45]
    },
}

class dataloader:

    def __init__(self, data1, data2, subsampling:int=1):
        # Quick and dirty mapping: assume range for each joint given exercise => map angle and then map from -1 to 1
        # For drifting, low-pass filter?

        # Create mappings
        self.sim2glovemap = {} # sim to glove ID map
        self.simmap = {}
        self.glovemap = {}
        for joint in sim2glove:
            if list(sim2glove[joint]["id"].values())[0] is not None:
                # Fix glove index with -1
                self.sim2glovemap[list(sim2glove[joint]["id"].keys())[0]] = list(sim2glove[joint]["id"].values())[0]-1
                self.simmap[list(sim2glove[joint]["id"].keys())[0]] = sim2glove[joint]["simmap"]

        # Load data
        self.data1 = scipy.io.loadmat(data1) # all data
        self.data2 = scipy.io.loadmat(data2) # calibrated gloves

        self.emg = self.data1["emg"][::subsampling, :]
        print(self.emg.shape)
        #plt.plot(self.emg[:,1])
        #plt.show()

        joints = self.data2["angles"] # Calibrated data
        #for joint in range(len(joints[0,:])):
        #    print(joint)
        #    plt.plot(joints[:,joint])
        #    plt.show()
        print(joints.shape)

        # TODO: find middle point using median of half of ordered points and use this as baseline?

        # Reorder data to be in same positions of sim space (others are ignored)
        joints[:, list(self.sim2glovemap.keys())] = joints[:, list(self.sim2glovemap.values())]
        #plt.plot(joints[:,list(self.simmap.keys())])
        #plt.show()

        # Apply mapping from calibrated angle to sim value
        for i in self.simmap:
            lmap = interp1d(self.simmap[i], [-1,1], fill_value="extrapolate")
            joints[:,i] = lmap(joints[:,i])

        # Clip data
        joints = np.clip(joints, -1, 1)
        #plt.plot(joints[:,list(self.simmap.keys())])
        #plt.show()

        # Only keep action values
        ACTION_SPACE_SIZE = 20
        self.pose = np.zeros((joints.shape[0],ACTION_SPACE_SIZE))
        self.pose[:, list(self.sim2glovemap.keys())] = joints[:, list(self.sim2glovemap.keys())]
        self.pose = self.pose[::subsampling, :]

        # Split data into trials (keep 5 for training, 1 for testing)
        # Each trial is an episode, call "redraw" to switch episode (picked at random)
        # Find rising and falling edges in restimulus (start and end of trials)
        restimulus = self.data1["restimulus"][::subsampling, :]
        edges = [(s!=restimulus[id-1])*((s>0) + -1*(s==0)) for id,s in enumerate(restimulus)]

        # Find indexes of edges
        indexes = [id for id,s in enumerate(edges) if s!=0]

        # Find ranges of samples for each trial
        trial_ranges = []
        last = -1
        for i in range(int(len(indexes)/2)-1):
            start = last+1
            end = int((indexes[2*i+1]+indexes[2*i+2])/2)
            trange = [start, end]
            trial_ranges.append(trange)
            last = end
        trial_ranges.append([last+1, len(edges)-1])

        #plt.plot(self.data1["restimulus"])
        #plt.plot(np.ones(trial_ranges[0][1]-trial_ranges[0][0]))
        #plt.show()

        print(self.emg.shape)
        print(self.pose.shape)

        # Gather trials
        trials = []
        for trial_range in trial_ranges:
            trial = np.concatenate((self.emg[trial_range[0]:trial_range[1], :], self.pose[trial_range[0]:trial_range[1], :]), axis=1)
            trials.append(trial)
        self.trials = trials

        # For each exercise, randomly select one trial as
        test_ids = np.zeros(int(len(trials)/6))

        for i in range(int(len(trials)/6)):
            test_ids[i] = np.random.randint(i*6,(i+1)*6)

        self.train_trials = [self.trials[id] for id in range(len(self.trials)) if id not in test_ids]
        self.test_trials = [self.trials[id] for id in range(len(self.trials)) if id in test_ids]


        self.current_train_trial = 0
        self.current_test_trial = 0
        self.set_mode()

    # Train functions
    def get_train_sample(self, index):
        # Sample is observation (EMG, Pose)
        return self.train_trials[self.current_train_trial][index]

    def get_num_train_samples(self):
        return len(self.train_trials[self.current_train_trial])

    def get_num_train_trials(self):
        return len(self.train_trials)

    # Test funtions
    def get_test_sample(self, index):
        # Sample is observation (EMG, Pose)
        return self.test_trials[self.current_test_trial][index]

    def get_num_test_samples(self):
        return len(self.train_trials[self.current_train_trial])

    def get_num_test_trials(self):
        return len(self.train_trials)

    def draw(self, trial):
        self.current_train_trial = trial
        self.current_test_trial = trial

    def set_mode(self, mode="train"):
        if mode == "train":
            self.get_sample = self.get_train_sample
            self.get_num_samples = self.get_num_train_samples
            self.get_num_trials = self.get_num_train_trials
        elif mode == "test":
            self.get_sample = self.get_test_sample
            self.get_num_samples = self.get_num_test_samples
            self.get_num_trials = self.get_num_test_trials

    def shuffle(self, test_ids):
        self.train_trials = [self.trials[id] for id in range(len(self.trials)) if id not in test_ids]
        self.test_trials = [self.trials[id] for id in range(len(self.trials)) if id in test_ids]
