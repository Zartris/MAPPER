import gym

from gym_game.envs import CommunicateEnv

if __name__ == '__main__':
    env = gym.make("SpeakAndLearn-v0", world_id=1, amount_of_bots=10)
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    print(states, actions)
