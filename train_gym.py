import argparse
import os
import time
from collections import deque
from functools import partial
from pathlib import Path

import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter
# from unityagents import UnityEnvironment

from DRL.agent.MA_TD3_agent import MATD3Agent, MATD3AgentConv
from DRL.agent.multi_agent import MultiAgent
from DRL.model.twin_ac_model import Actor, ActorConv
from DRL.replay_buffers.replay_buffer import ReplayBuffer
from utils import log
from gym_game.envs import CommunicateEnv


def eval_agent(game_env, agent, n_episodes=1000, max_t=1000, print_every=100, slow_and_pretty=True):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = game_env.reset(train_mode=not slow_and_pretty)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        agent.reset()  # Reset all agents
        score = np.zeros(num_agents)
        start_time = time.time()
        while True:
            actions = agent.act(states, add_noise=False)  # select an action (for each agent)
            env_info = game_env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            dones = env_info.local_done  # see if episode finished
            score += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            if np.any(dones):
                break
        duration = time.time() - start_time
        mean_score = np.mean(score)
        min_score = min(score)
        max_score = max(score)
        scores_deque.append(max_score)
        scores.append(mean_score)
        avg_score = np.mean(scores_deque)
        log_str = ('Episode {}'
                   '\tAverage Score: {:.2f} '
                   '\t current mean: {:.2f}'
                   '\t Min:{:.2f}'
                   '\tMax:{:.2f}'
                   '\tDuration:{:.2f}').format(i_episode, avg_score, mean_score, min_score, max_score, duration)
        print("\r" + log_str,
              end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    return scores


def train_agent(game_env, agent, action_size, n_episodes=1000, max_t=1000, file="", logging_folder="",
                log_every=5,
                print_every=100, warmups=0, slow_and_pretty=False):
    writer = SummaryWriter(str(logging_folder))
    scores_deque = deque(maxlen=print_every)
    scores = []
    log.file_append(file, "##Training stats:\n")
    logging_buffer = ""
    # Full buffer with random moves:
    for ep in range(1, warmups + 1):
        env_info = game_env.reset()  # reset the environment
        states = env_info.get_color_obs()  # get the current state (for each agent)
        for t in range(max_t):
            actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
            actions = np.clip(actions, 0, 4)  # all actions between -1 and 1
            env_info = env.step(actions)  # send all actions to the environment
            next_states = env_info.obs  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.add_to_memory(states, actions, dones, next_states, rewards)
            states = next_states
            env.render()
            if np.any(dones):
                break
    current_best = 0.1
    beaten = False
    for i_episode in range(1, n_episodes + 1):
        env_info = game_env.reset()  # reset the environment
        states = env_info.obs  # get the current state (for each agent)
        agent.reset()  # Reset all agents
        score = np.zeros(num_agents)
        start_time = time.time()
        steps = 0
        # for t in range(max_t):
        while True:
            actions = agent.act(states)  # select an action (for each agent)
            env_info = game_env.step(actions)  # send all actions to tne environment
            next_states = env_info.obs  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            agent.step(states, actions, rewards, next_states, dones)
            score += env_info.rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            env.render()
            steps += 1
            if np.any(dones):
                break
            if steps >= max_t:
                break
        duration = time.time() - start_time
        mean_score = np.mean(score)
        min_score = min(score)
        max_score = max(score)
        scores_deque.append(max_score)
        scores.append(mean_score)
        writer.add_scalar('score_per_episode', mean_score, i_episode - 1)
        avg_score = np.mean(scores_deque)
        writer.add_scalar('score_avg_over_100_episodes', avg_score, i_episode - 1)
        log_str = ('Episode {}'
                   '\tAverage Score: {:.2f} '
                   '\t current mean: {:.2f}'
                   '\t Min:{:.2f}'
                   '\tMax:{:.2f}'
                   '\tDuration:{:.2f},'
                   '\tStepCount: {:.2f} ').format(i_episode, avg_score, mean_score, min_score, max_score, duration,
                                                  steps)
        logging_buffer += "\t" + log_str + "\n"
        print("\r" + log_str,
              end="")
        if avg_score > round(current_best + 0.01, 3):
            log_str = "\nAt episode {} the current best {:.2f} has been beaten by {:.2f}, so we save the model".format(
                i_episode,
                current_best,
                avg_score)
            print(log_str)
            logging_buffer += log_str
            current_best = avg_score
            agent.save_all()
        if i_episode % log_every == 0:
            log.file_append(file, logging_buffer)
            logging_buffer = ""
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if i_episode >= 100 and avg_score > 0.5 and not beaten:
            beaten = True
            log_str = ('\nEnvironment solved in {:d} episodes!'.format(i_episode))
            logging_buffer += log_str
            print(log_str)
            log.file_append(file, logging_buffer)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)  # The seed for testing
    parser.add_argument("--max_timesteps", default=50, type=int)  # Max time per episode
    parser.add_argument("--episodes", default=4000, type=int)  # Number of episodes to train for
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for training [Tried, 256, 512, 1024]
    parser.add_argument("--buffer_size", default=int(1e5), type=int)  # Batch size for training [Tried, 1e5, 1e6, 2**20]
    parser.add_argument("--discount", default=0.99)  # Discount factor [Tried, 0.99, 0.995]
    parser.add_argument("--tau", default=1e-2)  # Soft update factor [Tried, 1e-2, 1e-3, 1e-4]
    parser.add_argument("--lr_actor", default=1e-4)  # Optimizer learning rate for the actor [Tried, 1e-3, 1e-4]
    parser.add_argument("--lr_critic", default=1e-3)  # Optimizer learning rate for the critic [Tried, 1e-3, 1e-4]
    parser.add_argument("--warmup_rounds", default=0)  # Optimizer learning rate for the critic
    parser.add_argument("--weight_decay", default=0)  # Optimizer learning rate for the critic [Tried, 0, 1e-6]
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--exploration_noise", default=0.3)  # Std of Gaussian exploration noise
    parser.add_argument("--noise_reduction_factor", default=0.999)  # Reducing the noise [Tried, 0.99, 0.999]
    parser.add_argument("--noise_scalar_init", default=2)  # initialise noise at start of each episode [Tried, 1, 2]
    parser.add_argument("--train_delay", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--steps_before_train", default=4, type=int)  # Steps taken between train calls.
    parser.add_argument("--train_iterations", default=2,
                        type=int)  # number of batches trained on per train call [Tried, 1, 2]
    parser.add_argument("--result_folder", default=os.path.join(os.getcwd(), "DRL/results"))
    parser.add_argument("--load_model_path", default="")  # If should load model: if "" don't load anything
    parser.add_argument("--eval", default=False, type=bool)  # If we only want to evaluate a model.
    parser.add_argument("--eval_load_best", default=False, type=bool)  # load best model (used by reviewers)
    parser.add_argument("--slow_and_pretty", default=True, type=bool)  # If we only want to evaluate a model.
    parser.add_argument("--number_of_agents", default=2, type=int)  # How many agents in the environment

    args = parser.parse_args()
    # Logging data:
    result_folder = Path(args.result_folder)
    if not result_folder.exists():
        result_folder.mkdir(parents=True)
    counter = 0
    test_folder = Path(result_folder, 'test' + str(counter))
    while test_folder.exists():
        counter += 1
        test_folder = Path(result_folder, 'test' + str(counter))
    test_folder.mkdir(parents=True)
    tb_logging = Path(test_folder, 'logging')
    if not tb_logging.exists():
        tb_logging.mkdir(parents=True)
    save_model_path = Path(test_folder)
    print(str(save_model_path))
    model_test_file = Path(test_folder, "model_test.md")

    log.log_hyper_para(file=model_test_file, args=args)

    env = CommunicateEnv(world_id=1, amount_of_bots=args.number_of_agents, slow_and_pretty=args.slow_and_pretty)

    #### 2. Examine the State and Action Spaces
    # number of agents
    num_agents = args.number_of_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = 1  # Dimension of each action
    action_val_high = 4
    action_val_low = 0
    print('Size of each action:', action_size)

    # examine the state space
    # reset the environment
    env_info = env.reset()
    states = env_info.obs
    state_size = states.shape[1:]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    #### 3. Take Random Actions in the Environment
    model_dir = args.load_model_path if args.load_model_path != "" else str(save_model_path)
    eval = args.eval
    if args.eval_load_best:
        model_dir = Path(os.getcwd(), "results", "solved")
        eval = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    replay_buffer_func = partial(ReplayBuffer, action_size=action_size, buffer_size=args.buffer_size,
                                 batch_size=args.batch_size, seed=args.seed)
    agents = []
    actor_func = partial(Actor, state_size=state_size, action_size=action_size, seed=args.seed, fc1_units=256,
                         fc2_units=128)
    for i in range(1, num_agents + 1):
        agents.append(MATD3Agent("MATD3Agent" + str(i),
                                 actor_func=actor_func,
                                 state_size=state_size,
                                 replay_buffer_func=replay_buffer_func,
                                 action_size=action_size,
                                 action_val_high=action_val_high,
                                 action_val_low=action_val_low,
                                 save_path=model_dir,
                                 seed=args.seed,
                                 train_delay=args.train_delay,
                                 steps_before_train=args.steps_before_train,
                                 train_iterations=args.train_iterations,
                                 discount=args.discount,
                                 tau=args.tau,
                                 lr_actor=args.lr_actor,
                                 lr_critic=args.lr_critic,
                                 policy_noise=args.policy_noise,
                                 noise_clip=args.noise_clip,
                                 exploration_noise=args.exploration_noise,
                                 noise_reduction_factor=args.noise_reduction_factor,
                                 noise_scalar_init=args.noise_scalar_init))
    agent = MultiAgent("MultiAgent",
                       agents=agents,
                       save_path=model_dir,
                       seed=args.seed)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    # Creating wrapper to handle multiple agents.
    # agent = MultiAgent(agents=agents, seed=seed, action_size=action_size, state_size=state_size)
    if not eval:
        if args.load_model_path != "":
            agent.load_all()
        scores = train_agent(game_env=env,
                             agent=agent,
                             action_size=action_size,
                             n_episodes=args.episodes,
                             max_t=args.max_timesteps,
                             file=str(model_test_file),
                             logging_folder=str(tb_logging),
                             warmups=args.warmup_rounds,
                             slow_and_pretty=args.slow_and_pretty)
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    else:
        agent.load_all()
        scores = eval_agent(game_env=env,
                            agent=agent,
                            n_episodes=args.episodes,
                            max_t=args.max_timesteps,
                            slow_and_pretty=args.slow_and_pretty)
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
