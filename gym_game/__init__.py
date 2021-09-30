from gym.envs.registration import register

register(
    id='SpeakAndLearn-v0',
    entry_point='gym_game.envs:CommunicateEnv',
    max_episode_steps=20000
)
