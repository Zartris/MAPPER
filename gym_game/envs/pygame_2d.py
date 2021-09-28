import ast

import cv2
import numpy as np
import queue
from colorhash import ColorHash

action_to_move = {0: "stay",
                  1: "up",
                  2: "down",
                  3: "left",
                  4: "right"}


def empty_room_world(world_size):
    # make borders:
    world = np.zeros(world_size)
    world[0, :] = 1.
    world[:, 0] = 1.
    world[world_size[0] - 1, :] = 1.
    world[:, world_size[1] - 1] = 1.
    return world


def get_random_free_position(world, world_size):
    x = np.random.randint(0, world_size[0])
    y = np.random.randint(0, world_size[1])
    # Check if there is any static object in the way
    if world[x, y] == 1.:
        x, y = get_random_free_position(world, world_size)
    return x, y


class PyGame2D:
    def __init__(self, world_id, world_size, amount_of_bots, vision_range):
        # Static infos
        self.world_id = world_id
        self.amount_of_bots = amount_of_bots
        self.world_static = empty_room_world(world_size)
        self.bot_vision_range = vision_range

        # Init goal positions
        goals = [get_random_free_position(self.world_static, world_size) for _ in range(amount_of_bots)]
        self.goal_map = {}
        self.init_goal_positions(goals)

        # Init Robots:
        self.bots = [Robot(i, self.world_static, self.bot_vision_range) for i in range(amount_of_bots)]
        self.init_bot_positions()

        self.crashed_dynamic = False
        self.crashed_static = False
        self.dynamic_crash_map = {}
        # Reward weights:
        self.crash_reward = -500
        self.goal_reward = 250
        self.move_reward = -1
        self.stay_reward = -0.5

        # Saved variables
        self.reward = [0 for _ in range(amount_of_bots)]
        self.last_actions = None

    def init_bot_positions(self):
        list_of_accepted_positions = []
        for bot in self.bots:
            if (bot.x, bot.y) in list_of_accepted_positions:
                while (bot.x, bot.y) in list_of_accepted_positions:
                    bot.place_at_random_position()
            list_of_accepted_positions.append((bot.x, bot.y))

    def init_goal_positions(self, goals):
        list_of_accepted_positions = []
        for i, position in enumerate(goals):
            if position in list_of_accepted_positions:
                while position in list_of_accepted_positions:
                    position = get_random_free_position(self.world_static, self.world_static.shape[:2])
            list_of_accepted_positions.append(position)

        for position in list_of_accepted_positions:
            self.goal_map[str(position)] = False  # Visited

    def actions(self, actions):
        self.last_actions = actions
        for i, bot in enumerate(self.bots):
            bot.action(action_to_move[actions[i]])

    def compute_action_reward(self, action):
        if action == "up":
            return self.move_reward
        elif action == "down":
            return self.move_reward
        elif action == "left":
            return self.move_reward
        elif action == "right":
            return self.move_reward
        else:  # Stay
            return self.stay_reward

    def evaluate(self):
        """
        Computes reward for each bot
        :return:
        """
        self.reward = [0 for _ in range(amount_of_bots)]

        # Building crash map more memory used, but faster to compute:
        self.dynamic_crash_map = {}
        for bot in self.bots:
            if str(bot.position()) in self.dynamic_crash_map:
                self.dynamic_crash_map[str(bot.position())].append(bot.id)
            else:
                self.dynamic_crash_map[str(bot.position())] = [bot.id]

        for i, bot in enumerate(self.bots):
            # Evaluate for crash
            self.reward[i] += self.check_for_bot_crash(bot, self.dynamic_crash_map, verbose=True)

            # Check if we reached a new goal:
            if str(bot.position()) in self.goal_map and not self.goal_map[str(bot.position())]:
                # add goal reached:
                self.goal_map[str(bot.position())] = True
                self.reward[i] += self.goal_reward

            # Compute action cost / reward:
            self.reward[i] += self.compute_action_reward(action_to_move[self.last_actions[i]])
        return self.reward

    def is_done(self):
        done = False
        # Cases where it is done:
        # 1. One of the bot crashed
        done = done or self.crashed_static or self.crashed_dynamic
        # 2. all targets is reached
        # 3. Times up.
        # 4. .....
        return done

    def observe(self):
        # 1. If bots are in vision, copy share information
        for bot in self.bots:
            bot.do_observation(self.bots)
        return 0

    def view(self):
        # Render the view.
        canvas_gray = np.copy(self.world_static)
        canvas_gray *= 255
        canvas_gray = 255 - canvas_gray
        canvas_color = np.stack((canvas_gray,) * 3, axis=-1).astype(np.uint8)
        # Draw goals first:
        for goal_point_str in self.goal_map.keys():
            goal_p = ast.literal_eval(goal_point_str)
            canvas_color[goal_p[0], goal_p[1]] = (0, 255, 0)

        list_of_local_obs = []
        # Draw bots after:
        for i, bot in enumerate(self.bots):
            color = ColorHash(i * 10)
            canvas_color[bot.x, bot.y] = color.rgb
            list_of_local_obs.append(bot.vertically_combined_local_observations())
        return canvas_color.astype(np.uint8), list_of_local_obs

    def check_for_bot_crash(self, bot, dynamic_crash_map, verbose=False):
        reward = 0
        if len(dynamic_crash_map[str(bot.position())]) > 1:
            if verbose:
                print(
                    f"Crash in world_id: {self.world_id}\n"
                    f"Robot_{bot.id} crashed into another bot [{str(dynamic_crash_map[str(bot.position())])}]")
            reward += self.crash_reward
            self.crashed_dynamic = True
        if self.world_static[bot.x, bot.y] == 1.:
            if verbose:
                print(f"Crash in world_id: {self.world_id}, robot_{bot.id} crashed into static environment")
            reward += self.crash_reward
            self.crashed_static = True
        return reward


class Robot:
    def __init__(self, id, world, vision_range, trail_length=3):
        self.id = id
        self.x = 0
        self.y = 0
        self.trail_length = trail_length
        self.trail = []
        self.vision_range = vision_range

        self.world_map = world
        self.world_size = world.shape[:2]
        self.global_goal_trajectory = np.zeros(world.shape[:2])

        self.local_observations = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))  # Local map
        self.local_dynamic_tracking = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))
        self.local_goal_trajectory = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))

        self.place_at_random_position()
        for i in range(trail_length):
            self.trail.append(self.position())

    def place_at_random_position(self):
        self.x, self.y = get_random_free_position(self.world_map, self.world_size)
        return self.position()

    def action(self, action):
        # adding trail before overwriting pos:
        self.add_trail(self.x, self.y)
        if action == "up":
            self.y += 1
        elif action == "down":
            self.y -= 1
        elif action == "left":
            self.x -= 1
        elif action == "right":
            self.x += 1

    def add_trail(self, x, y):
        del self.trail[0]
        self.trail.append((x, y))

    def position(self):
        return self.x, self.y

    def do_observation(self, dynamic_objects):
        # Static local observations
        # TODO:: Not taking into account that some might be close to border
        self.local_observations = self.world_map[
                                  self.x - self.vision_range: self.x + self.vision_range + 1,
                                  self.y - self.vision_range: self.y + self.vision_range + 1
                                  ]

        w, h = self.local_observations.shape[:2]
        # Add the dynamic objects:
        for do in dynamic_objects:
            if do.id == self.id:
                continue
            x, y = do.position()
            lx, ly = (x - self.x) + vision_range, (y - self.y) + vision_range
            if 0 <= lx <= w and 0 <= ly <= h:
                self.local_observations[lx, ly] = 0.5
        debug = 0
        return self.local_observations, self.local_dynamic_tracking, self.local_goal_trajectory

    def vertically_combined_local_observations(self):
        return np.vstack((self.local_observations, self.local_dynamic_tracking, self.local_goal_trajectory))


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    world_size = (40, 40)
    canvas_size = (600, 600)
    amount_of_bots = 5
    vision_range = 10

    game = PyGame2D(1, world_size, amount_of_bots, vision_range)


    def step(pygame, actions):
        pygame.actions(actions)
        obs = pygame.observe()
        reward = pygame.evaluate()
        done = pygame.is_done()
        return obs, reward, done, {}


    running = True
    steps = 0
    while running:
        steps += 1
        actions = [np.random.randint(0, 5) for _ in range(amount_of_bots)]
        obs, reward, done, notes = step(game, actions)
        running = not done
        print(f"steps: {steps}\n"
              f"obs: {obs}\n"
              f"reward: {reward}\n"
              f"done: {done}\n"
              f"notes: {str(notes)}\n\n")
        world, list_of_local_views = game.view()
        world_scaled = cv2.resize(world, (0, 0), fx=canvas_size[0] / world_size[0], fy=canvas_size[1] / world_size[1])
        cv2.imshow("world", world_scaled)
        for i, img in enumerate(list_of_local_views):
            cv2.imshow(i, img)
        if cv2.waitKey(0):
            cv2.destroyAllWindows()
