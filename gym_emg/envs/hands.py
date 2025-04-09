import numpy as np
import os
from gym import utils, error
from gym.envs.robotics.utils import robot_get_obs
from gym.envs.robotics import rotations, hand_env

from dexterous_gym.core.two_hand_robot_env import RobotEnv

import pathlib

TWO_HAND_XML = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets/hand/2hands.xml')
print(TWO_HAND_XML)

class BaseHandEnv(RobotEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)

        self.n_actions = 40 # Two static hands with only joints moving
        n_substeps=20
        initial_qpos = {}
        relative_control = False   

        """
        super(RobotEnv, self).__init__(
            model_path=TWO_HAND_XML, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos
        )
        """
        RobotEnv.__init__(self,
            model_path=TWO_HAND_XML, n_substeps=n_substeps, n_actions=self.n_actions, initial_qpos=initial_qpos
        )

    # TwoHandsEnv methods
    # ----------------------------
    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_centre = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_centre + action*actuation_range
        self.sim.data.ctrl[:] = np.clip(self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1])

    # RobotEnv methods
    # ----------------------------
    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

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
        return True # Return True?

    def _sample_goal(self):
        # Goal is current position for joints
        goal =  np.array([0., 0., 1.])
        return goal

    def _render_callback(self):
        # Render sim
        self.sim.forward()

    def _get_obs(self):
        # Observe current robot position AND EMG samples
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        #achieved_goal = self._get_achieved_goal().ravel()  # this contains the current hand position?? (achieved goal??)
        achieved_goal = np.concatenate([robot_qpos, robot_qvel])
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
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
    def __init__(self, direction=1, alpha=1.0):
        self.direction = direction #-1 or 1
        self.alpha = alpha
        super(TwoHands, self).__init__()
        #self.bottom_id = self.sim.model.site_name2id("object:bottom")
        #self.top_id = self.sim.model.site_name2id("object:top")
        self._max_episode_steps = 2000
        self.observation_space = self.observation_space["observation"]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = {}
        reward = self.compute_reward()
        return obs["observation"], reward, done, info

    def compute_reward(self):
        # Reward is based on current position vs desired position
        reward_1 = 0 # Used to be based on Pen position
        reward_2 = 0 # Used to be based on Pen velocity
        return self.alpha * reward_2 + reward_1

    def reset(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = np.array([10000,10000,10000,0,0,0,0]) #hide offscreen
        obs = self._get_obs()["observation"]
        return obs