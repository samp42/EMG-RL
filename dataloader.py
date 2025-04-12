import scipy.io
import os
import numpy as np
from scipy.interpolate import interp1d

# CyberGlove II mapping: https://www.researchgate.net/figure/Cyberglove-II-device-and-the-placement-of-the-22-sensors_fig1_338483027 
sim2glove = {
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

    "Ta": { # Thumb abduction (proximal)
        "id": {17:None},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "TM1": { # Thumb metacarpal 1 (to/from palm)
        "id": {15:1},
        "glovemap": [0,100],
        "simmap": [0, 90] # OK
    },
    "TM2": { # Thumb metacarpal 2 (lateral to palm)
        "id": {16:4},
        "glovemap": [0,100],
        "simmap": [0, 45] # OK
    },
    "TP": { # Thumb proximal
        "id": {18:2},
        "glovemap": [0,100],
        "simmap": [0, 90]
    },
    "Tb": { # Thumb ptit boutte
        "id": {19:3}, # OK
        "glovemap": [100,0],
        "simmap": [0, 90]
    },
}


class dataloader:

    def __init__(self, folderpath):
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
                self.glovemap[list(sim2glove[joint]["id"].values())[0]-1] = sim2glove[joint]["glovemap"]
                self.simmap[list(sim2glove[joint]["id"].keys())[0]] = sim2glove[joint]["simmap"]

        # Load data
        # TODO: combine all together into one big signal?
        import matplotlib.pyplot as plt
        file = "/home/jacobyroy/Desktop/COMP579/s3/S3_E2_A1.mat"
        self.data = scipy.io.loadmat(file)
        joints = self.data["glove"]

        # Find min/max of samples (known to be angles of mapping)
        mins = np.min(joints, axis=0)
        maxs = np.max(joints, axis=0)
        print(mins[list(self.glovemap.keys())])
        print(maxs[list(self.glovemap.keys())])
        #plt.plot(joints[:,list(self.glovemap.keys())])
        #plt.show()

        # Apply mapping from raw glove data to angle range
        for i in self.glovemap:
            lmap = interp1d([mins[i], maxs[i]], self.glovemap[i], fill_value="extrapolate")
            joints[:,i] = lmap(joints[:,i])
        #plt.plot(joints[:,list(self.glovemap.keys())])
        #plt.show()

        # Reorder data to be in same positions of sim space (others are ignored)
        joints[:, list(self.sim2glovemap.keys())] = joints[:, list(self.sim2glovemap.values())]
        #plt.plot(joints[:,list(self.simmap.keys())])
        #plt.show()

        # Apply mapping from angle to sim
        for i in self.simmap:
            lmap = interp1d(self.simmap[i], [-1,1], fill_value="extrapolate")
            joints[:,i] = lmap(joints[:,i])
        #plt.plot(joints[:,list(self.simmap.keys())])
        #plt.show()

        # Clip data
        joints = np.clip(joints, -1, 1)
        #plt.plot(joints[:,list(self.simmap.keys())])
        #plt.show()

        # Only keep action values
        ACTION_SPACE_SIZE = 20
        self.pose = np.zeros((joints.shape[0],ACTION_SPACE_SIZE))
        self.pose[:, list(self.sim2glovemap.keys())] = joints[:, list(self.sim2glovemap.keys())]

        # Find E2, which contains exercises with full range

        # Find E2 range for all channels

    def get_sample(self, index):
        # Sample is observation (EMG, Pose)
        return self.pose[index,:]
        