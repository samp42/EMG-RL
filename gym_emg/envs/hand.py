import numpy as np
import os
from gymnasium import utils, error
import mujoco
# from gym.envs.robotics.hand.manipulate import ManipulateEnv, HandPenEnv
# from gym.envs.robotics.utils import robot_get_obs
# from gym.envs.robotics import rotations, hand_env
# from gymnasium_robotics.utils import ro
from gymnasium_robotics.utils.mujoco_utils import robot_get_obs
import gymnasium_robotics.envs.shadow_dexterous_hand.hand_env as hand_env
# import gym.envs.robotics.hand.manipulate as module

# L = "/home/jacobyroy/miniconda3/envs/emg_rl/lib/python3.8/site-packages/gym/envs/robotics/assets"


MANIPULATE_PEN_XML = os.path.join('hand', 'manipulate_pen.xml')

import pathlib

MANIPULATE_HAND_XML = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets/hand/manipulate_hand.xml')
print(MANIPULATE_HAND_XML)

hand_env.HandEnv = hand_env.get_base_hand_env(hand_env.MujocoRobotEnv)

class BaseHandEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)

        n_substeps=20
        initial_qpos = {}
        relative_control = False        
        hand_env.HandEnv.__init__(
                    self, model_path=MANIPULATE_HAND_XML, n_substeps=n_substeps, initial_qpos=initial_qpos,
                    relative_control=relative_control)

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
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco.MujocoException:
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


class SingleHand(BaseHandEnv):
    def __init__(self, direction=1, alpha=1.0):
        self.direction = direction #-1 or 1
        self.alpha = alpha
        super(SingleHand, self).__init__()
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