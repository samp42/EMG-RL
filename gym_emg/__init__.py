from gym.envs.registration import register


register(
    id='gym_emg/SingleHand-v0',
    entry_point='gym_emg.envs:SingleHand',
    max_episode_steps=300,
)