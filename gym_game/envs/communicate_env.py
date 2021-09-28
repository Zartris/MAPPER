import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_game.envs import PyGame2D


class CommunicateEnv(gym.Env):
    def __init__(self, amount_of_bots):
        self.seed()

        self.amount_of_bots = amount_of_bots
        self.world_size = 100
        self.vision_range = 10
        self.pygame = PyGame2D(world_size=(self.world_size, self.world_size),
                               amount_of_bots=self.amount_of_bots,
                               vision_range=self.vision_range)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0,
                                            high=5,
                                            shape=(
                                                self.vision_range * 2 + 1,
                                                self.vision_range * 2 + 1,
                                                amount_of_bots),
                                            dtype=np.int)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        self.pygame.actions(actions)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        return obs, reward, done, {}

    def reset(self):
        del self.pygame
        self.pygame = PyGame2D(world_size=(self.world_size, self.world_size),
                               amount_of_bots=self.amount_of_bots,
                               vision_range=self.vision_range)
        obs = self.pygame.observe()
        return obs

    def render(self, mode="human", close=False):
        self.pygame.view()
