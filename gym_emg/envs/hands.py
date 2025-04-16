import mujoco_py
import numpy as np
import os
from gym import utils, error, spaces
from gym.envs.robotics.utils import robot_get_obs
from gym.envs.robotics import rotations, hand_env
from dexterous_gym.core.two_hand_robot_env import RobotEnv
import pathlib

from .dataloader import dataloader

TWO_HAND_XML = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets/hand/2hands.xml')
print(TWO_HAND_XML)

class BaseHandEnv(RobotEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse', n_substeps:int=20, datapath="~", subject=1, exercise=2, subsampling:int=1):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)

        # Load subject data
        print(f"Subject {subject}, Exercise {exercise}")
        data1_path = pathlib.Path(f"{datapath}/s{subject}/S{subject}_E{exercise}_A1.mat")
        data2_path = pathlib.Path(f"{datapath}/s_{subject+67}_angles/s_{subject+67}_angles/S{subject+67}_E{exercise}_A1.mat")
        self.loader = dataloader(data1_path, data2_path, subsampling=subsampling)
        self.sample_counter = 0
        #self._max_episode_steps = None # Run indefinitely (set when registering environment)

        self.n_actions = 40 # Two static hands with only joints moving
        n_substeps=n_substeps
        initial_qpos = {}
        relative_control = False   

        #super(RobotEnv, self).__init__(
        #    model_path=TWO_HAND_XML, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos
        #)
        RobotEnv.__init__(self,
            model_path=TWO_HAND_XML, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos, 
        )

        # Define action space and observation space
        n_actions = 20
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    # TwoHandsEnv methods
    # ----------------------------
    def _set_action(self, action):
        #assert action.shape == (self.n_actions,)
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_centre = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_centre + action*actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    # RobotEnv methods
    # ----------------------------
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(40))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return True 

    def _sample_goal(self):
        # TODO: unused, but required by super class
        goal =  np.zeros(1)
        return goal

    def _render_callback(self):
        # Render sim
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim) # Dynamics of both hands
        obs = self.loader.get_sample(self.sample_counter) # Current sample (EMG + Desired Pose)

        observation = np.concatenate([robot_qpos[24::], robot_qvel[24::], obs[0:16]]) # Controlled hand dynamics + EMG

        return {
            'observation': observation.copy(),
            'achieved_goal': observation.copy(), # unused
            'desired_goal': self.goal.ravel().copy(), # unused
        }

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('robot0:palm')
        middle_id = self.sim.model.site_name2id('centre-point')
        # lookat = self.sim.data.body_xpos[body_id]
        lookat = self.sim.data.site_xpos[middle_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180.0
        self.viewer.cam.elevation = -55.0


class TwoHands(BaseHandEnv):
    def __init__(self, direction=1, alpha=1.0, datapath="~", n_substeps:int=20, subject=1, exercise=2, subsampling:int=1):
        self.direction = direction #-1 or 1
        self.alpha = alpha
        super(TwoHands, self).__init__(datapath=datapath, n_substeps=n_substeps, subject=subject, exercise=exercise, subsampling=subsampling)
        #self.bottom_id = self.sim.model.site_name2id("object:bottom")
        #self.top_id = self.sim.model.site_name2id("object:top")
        self.observation_space = self.observation_space["observation"]

    def step(self, action):
        # Action should only be for controlled hand, not reference
        ref_action = self.loader.get_sample(self.sample_counter)[16::] # Get hand pose reference
        self.sample_counter += 1
        done = False
        if self.sample_counter >= self.loader.get_num_samples()-1:
            done = True

        #print(f"Num samples: {self.sample_counter}, total: {self.loader.get_num_samples()}")
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ref_action = np.clip(ref_action, self.action_space.low, self.action_space.high)
        action = np.concatenate((action, ref_action)) # Need to concatenate after due to action space size
        self.action = action
        self._set_action(action)
        self.sim.step()

        self._step_callback() # not implemented

        obs = self._get_obs()
        info = {}
        reward = self.compute_reward()
        return obs["observation"], reward, done, info

    def compute_reward(self):
        # Reward is based on current position vs desired position (could also add penalty if static when it shouldnt, but would work better in multi-goal)

        # TODO: for now reward is negative of norm between target vs current
        # https://ras.papercept.net/images/temp/IROS/files/0530.pdf
        diff = self.action[0:int(self.n_actions/2)] - self.action[int(self.n_actions/2)::]
        reward = -np.linalg.norm(diff)
        return self.alpha * reward

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = np.zeros(1) # unused but required
        obs = self._get_obs()["observation"]
        return obs