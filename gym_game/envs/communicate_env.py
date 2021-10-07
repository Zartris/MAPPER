import cv2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_game.envs.pygame_2d import PyGame2D, EnvInfo
from image_processing.improc import hconcat_resize_min


class CommunicateEnv(gym.Env):
    def __init__(self, world_id, amount_of_bots, slow_and_pretty=False):
        self.seed()
        self.world_id = world_id
        self.amount_of_bots = amount_of_bots
        self.world_size = 20
        self.canvas_size = 600
        self.vision_range = 10
        self.pygame = PyGame2D(world_id=world_id,
                               world_size=(self.world_size, self.world_size),
                               amount_of_bots=self.amount_of_bots,
                               vision_range=self.vision_range)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0,
                                            high=1.,
                                            shape=(amount_of_bots,
                                                   self.vision_range * 2 + 1,
                                                   self.vision_range * 2 + 1),
                                            dtype=np.float)
        self.slow_and_pretty = slow_and_pretty

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        self.pygame.actions(actions)
        obs = self.pygame.observe()  # list
        reward = self.pygame.evaluate()
        dones = self.pygame.is_done()
        return EnvInfo(obs, reward, dones, {})

    def reset(self, slow_and_pretty=False):

        del self.pygame
        self.pygame = PyGame2D(world_id=self.world_id,
                               world_size=(self.world_size, self.world_size),
                               amount_of_bots=self.amount_of_bots,
                               vision_range=self.vision_range)
        obs = self.pygame.observe()
        return EnvInfo(obs)

    def render(self, mode="human", close=False):
        if self.slow_and_pretty:
            world, list_of_local_views = self.pygame.view()
            world_scaled = cv2.resize(world, (0, 0), fx=self.canvas_size / self.world_size,
                                      fy=self.canvas_size / self.world_size,
                                      interpolation=cv2.INTER_NEAREST)
            showInMovedWindow("World", world_scaled, 2020, 20)
            list_of_local_views_resized = []
            for i, local_view in enumerate(list_of_local_views):
                local_view_resized = cv2.resize(local_view, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
                list_of_local_views_resized.append(local_view_resized)
            local_view_combined = hconcat_resize_min(list_of_local_views_resized, cv2.INTER_NEAREST)
            # img = np.pad(img, pad_width=2, constant_values=0.5)
            showInMovedWindow("Local view combined", local_view_combined, 3000, 20)

            if cv2.waitKey(1):
                pass


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)  # Move it to (x,y)
    cv2.imshow(winname, img)
