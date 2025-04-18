from gymnasium.envs.registration import register

register(
    id='gym_emg/SingleHand-v0',
    entry_point='gym_emg.envs:SingleHand',
    #max_episode_steps=300,
)

# register(
#     id='gym_emg/TwoHands-v0',
#     entry_point='gym_emg.envs:TwoHands',
#     #max_episode_steps=300,
# )