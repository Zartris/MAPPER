from gym.envs.registration import register

register(
    id='speak_and_learn_game',
    entry_point='gym_game.envs:CommunicateEnv',
    max_episode_steps=20000
)
