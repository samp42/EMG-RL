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

        """
        # Load data
        # TODO: combine all together into one big signal?
        import matplotlib.pyplot as plt
        file = "/home/jacobyroy/Desktop/COMP579/s2/S2_E2_A1.mat"
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
        """

        # Load data
        # TODO: combine all together into one big signal?
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

    def get_sample(self, index):
        # Sample is observation (EMG, Pose)
        return np.concatenate((self.emg[index,:], self.pose[index,:]))
        
    def get_num_samples(self):
        return len(self.pose[:,0])