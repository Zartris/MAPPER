import ast

import cv2
import numpy as np
from colorhash import ColorHash

from image_processing.improc import vconcat_resize_min, hconcat_resize_min, add_text


class EnvInfo:
    def __init__(self, obs, rewards=[], local_done=[], notes={}):
        self.obs = obs
        self.rewards = rewards
        self.local_done = local_done
        self.notes = notes

    def get_color_obs(self):
        obs = np.array(self.obs)
        # obs = np.expand_dims(obs, 1)
        obs = np.stack((obs,) * 3, axis=1)
        return obs

    def get_obs(self):
        obs = np.array(self.obs)
        return np.expand_dims(obs, 1)


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
        # Padding so vision is for filled outside borders
        self.world_static = np.pad(self.world_static, vision_range + 1, constant_values=1.)
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
        self.crash_reward = -50
        self.goal_reward = 25
        self.move_reward = -0.1
        self.stay_reward = -0.1

        # Mini game:
        self.step_out_of_goal_reward = -250
        self.reached_a_goal = False

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
        actions = actions.squeeze().astype(np.int8)
        self.last_actions = actions
        for action, bot in zip(actions, self.bots):
            bot.action(action_to_move[action])

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
        # Reset bot rewards:
        for bot in self.bots:
            bot.reward = 0

        # Building crash map more memory used, but faster to compute:
        self.dynamic_crash_map = {}
        for bot in self.bots:
            if str(bot.position()) in self.dynamic_crash_map:
                self.dynamic_crash_map[str(bot.position())].append(bot.id)
            else:
                self.dynamic_crash_map[str(bot.position())] = [bot.id]

        for action, bot in zip(self.last_actions, self.bots):
            # Evaluate for crash
            bot.reward += self.check_for_bot_crash(bot, self.dynamic_crash_map, verbose=True)

            # Check if we reached a new goal:
            if str(bot.position()) in self.goal_map and not self.goal_map[str(bot.position())]:
                # add goal reached:
                self.goal_map[str(bot.position())] = True
                bot.reward += self.goal_reward

            # Compute action cost / reward:
            bot.reward += self.compute_action_reward(action_to_move[action])

        self.reward = [bot.reward for bot in self.bots]
        return self.reward

    def is_done(self):
        # Cases where it is done:
        # 1. One of the bot crashed
        dones = [bot.is_done for bot in self.bots]
        # 2. all targets is reached
        goals_hit = list(self.goal_map.values())
        if np.all(goals_hit):
            dones = [True for _ in self.bots]
        # 3. Times up.
        # 4. .....
        return dones

    def observe(self):
        # 1. If bots are in vision, copy share information
        obs = []
        for bot in self.bots:
            local_bot_obs = bot.do_observation(self.bots, self.goal_map)
            obs.append(local_bot_obs)
        obs = np.stack(obs, axis=0)
        return obs

    def view(self):
        # Render the view.
        canvas_gray = np.copy(self.world_static)
        canvas_gray *= 255
        # invert to make solid black and free space white
        canvas_gray = 255 - canvas_gray
        canvas_color = np.stack((canvas_gray,) * 3, axis=-1).astype(np.uint8)
        # Draw goals first:
        for goal_point_str in self.goal_map.keys():
            goal_p = ast.literal_eval(goal_point_str)
            canvas_color[goal_p[1], goal_p[0]] = (0, 255, 0)

        list_of_local_obs = []
        # Draw bots after:
        for i, bot in enumerate(self.bots):
            color = ColorHash(i * 10)
            canvas_color[bot.y, bot.x] = color.rgb
            list_of_local_obs.append(bot.get_view_color().astype(np.uint8))

        # invert to make solid black and free space white

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
            bot.is_done = True
        if self.world_static[bot.x, bot.y] == 1.:
            if verbose:
                print(f"Crash in world_id: {self.world_id}, robot_{bot.id} crashed into static environment")
            reward += self.crash_reward
            self.crashed_static = True
            bot.is_done = True
        return reward


class Robot:
    def __init__(self, id, world, vision_range, trail_length=5):
        self.id = id
        self.x = 0
        self.y = 0
        self.trail_length = trail_length
        self.trail_decay = 1 / trail_length
        self.trail = []
        self.vision_range = vision_range

        # Check if we want to pad here or do it later:
        self.world_map = world
        # self.world_map_padded = np.pad(self.world_map, vision_range, constant_values=1.)
        self.world_size = world.shape[:2]
        self.global_goal_trajectory = np.zeros(world.shape[:2])

        self.local_observations = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))  # Local map
        self.local_dynamic_tracking = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))
        self.local_goal_trajectory = np.zeros((vision_range * 2 + 1, vision_range * 2 + 1))
        self.last_action = "stay"

        self.place_at_random_position()
        for i in range(trail_length):
            self.trail.append(self.position())

        # Holding state:
        self.is_done = False
        self.reward = 0

    def place_at_random_position(self):
        self.x, self.y = get_random_free_position(self.world_map, self.world_size)
        return self.position()

    def action(self, action):
        # adding trail before overwriting pos:
        self.add_trail(self.x, self.y)
        if action == "up":
            self.y -= 1
        elif action == "down":
            self.y += 1
        elif action == "left":
            self.x -= 1
        elif action == "right":
            self.x += 1
        self.last_action = action

    def add_trail(self, x, y):
        del self.trail[0]
        self.trail.append((x, y))

    def position(self):
        return self.x, self.y

    def do_observation(self, dynamic_objects, goal_map: dict):
        # Static local observations
        self.local_observations = np.copy(self.world_map[
                                          self.y - self.vision_range: self.y + self.vision_range + 1,
                                          self.x - self.vision_range: self.x + self.vision_range + 1
                                          ])
        # Shift according to last action:
        self.local_dynamic_tracking = self.roll_map_based_on_action(self.local_dynamic_tracking)
        # Decay one timestep
        self.local_dynamic_tracking = np.clip(self.local_dynamic_tracking - self.trail_decay, 0., 1.)

        self.local_goal_trajectory = np.zeros(self.local_goal_trajectory.shape)  # Clear last view

        w, h = self.local_observations.shape[:2]
        # Add the dynamic objects:
        for do in dynamic_objects:
            x, y = do.position()
            lx, ly, inside_range = self.is_inside_vision_range(x, y, w, h)
            if inside_range:
                if do.id == self.id and self.local_observations[lx, ly] != 0.5:
                    self.local_observations[ly, lx] = 0.75
                else:
                    self.local_observations[ly, lx] = 0.5
                # Dynamic trails
                self.local_dynamic_tracking[ly, lx] = 1.0
        # Goal tracking
        for goal_point_str, visit in goal_map.items():
            if not visit:
                goal_p = ast.literal_eval(goal_point_str)
                x, y = goal_p
                lx, ly, inside_range = self.is_inside_vision_range(x, y, w, h)
                if inside_range:
                    self.local_goal_trajectory[ly, lx] = 1.
        return self.local_observations, self.local_dynamic_tracking, self.local_goal_trajectory

    def is_inside_vision_range(self, x, y, w, h):
        lx, ly = (x - self.x) + self.vision_range, (y - self.y) + self.vision_range
        return lx, ly, 0 <= lx < w and 0 <= ly < h

    def vertically_combined_local_observations(self, invert_colors=False):
        # Pad all imaged to separate them:
        list_to_resize = [np.pad(img, 1, constant_values=0.5) for img in
                          [self.local_observations, self.local_dynamic_tracking, self.local_goal_trajectory]]
        if invert_colors:
            list_to_resize = [1. - img for img in list_to_resize]
        return vconcat_resize_min(list_to_resize, cv2.INTER_NEAREST)

    def get_view_color(self):
        # Invert color:
        local_observations_color = 1. - self.local_observations
        local_dynamic_tracking_color = 1. - self.local_dynamic_tracking
        local_goal_trajectory_color = 1. - self.local_goal_trajectory

        # Convert to colors:
        local_observations_color = cv2.cvtColor((local_observations_color * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        local_dynamic_tracking_color = cv2.cvtColor((local_dynamic_tracking_color * 255).astype(np.uint8),
                                                    cv2.COLOR_GRAY2RGB)
        local_goal_trajectory_color = cv2.cvtColor((local_goal_trajectory_color * 255).astype(np.uint8),
                                                   cv2.COLOR_GRAY2RGB)
        move_text_view = (np.ones((local_goal_trajectory_color.shape)) * 255).astype(np.uint8)
        # Add special view Colors:
        local_observations_color[self.vision_range, self.vision_range] = (0, 0, 255)
        # Writing action taken:
        add_text(move_text_view, self.last_action, anchor=(0, 10), fontScale=0.2, thickness=1)
        # Pad to separate views:
        pad_color = (100, 100, 100)
        if self.world_map[self.x, self.y] == 1 or self.local_observations[self.vision_range, self.vision_range] != 0.75:
            pad_color = (0, 0, 255)
        list_to_resize = [local_observations_color, local_dynamic_tracking_color, local_goal_trajectory_color,
                          move_text_view]
        list_to_resize = [cv2.copyMakeBorder(img, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT,
                                             value=pad_color) for img in list_to_resize]
        return vconcat_resize_min(list_to_resize, cv2.INTER_NEAREST)

    def roll_map_based_on_action(self, map):
        new_map = np.copy(map)
        if self.last_action == "up":
            new_map = np.roll(new_map, 1, axis=0)
            new_map[0, :] = 0
        elif self.last_action == "down":
            new_map = np.roll(new_map, -1, axis=0)
            new_map[-1, :] = 0
        elif self.last_action == "left":
            new_map = np.roll(new_map, 1, axis=1)
            new_map[:, 0] = 0
        elif self.last_action == "right":
            new_map = np.roll(new_map, -1, axis=1)
            new_map[:, -1] = 0
        return new_map


if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    world_size = (40, 40)
    canvas_size = (600, 600)
    amount_of_bots = 5
    vision_range = 10

    game = PyGame2D(1, world_size, amount_of_bots, vision_range)


    def step(game, actions):
        game.actions(actions)
        obs = game.observe()  # list
        reward = game.evaluate()
        dones = game.is_done()
        return EnvInfo(obs, reward, dones, {})


    running = True
    steps = 0
    while running:
        steps += 1
        actions = [np.random.randint(0, 5) for _ in range(amount_of_bots)]
        obs, reward, done, notes = step(game, np.array(actions))
        running = not np.any(done)
        print(f"steps: {steps}\n"
              f"obs: {obs}\n"
              f"reward: {reward}\n"
              f"done: {done}\n"
              f"notes: {str(notes)}\n\n")
        world, list_of_local_views = game.view()
        world_scaled = cv2.resize(world, (0, 0), fx=canvas_size[0] / world_size[0], fy=canvas_size[1] / world_size[1],
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow("world", world_scaled)
        list_of_local_views_resized = []
        for i, local_view in enumerate(list_of_local_views):
            local_view_resized = cv2.resize(local_view, dsize=(0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            list_of_local_views_resized.append(local_view_resized)
        local_view_combined = hconcat_resize_min(list_of_local_views_resized, cv2.INTER_NEAREST)
        # img = np.pad(img, pad_width=2, constant_values=0.5)
        cv2.imshow("Local view combined", local_view_combined)

        if cv2.waitKey(0):
            cv2.destroyAllWindows()
