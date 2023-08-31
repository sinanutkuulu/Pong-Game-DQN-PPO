#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

import time
import numpy as np
import collections
from Environments import PongEnvironment
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt


torch.manual_seed(0)
DEFAULT_ENV_NAME = "PONG-ENV"
MEAN_REWARD_BOUND = 90.0

# HYPERPARAMETERS
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_STEPS = 100
REPLAY_START_SIZE = 10000
EPSILON_DECAY_LAST_STEP = 10000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


def plot_learning_curve(rewards):

    # Plot the actual rewards
    window_size = 100  # size of the window for the moving average
    avg_rewards = moving_average(rewards, window_size)
    plt.plot(range(window_size - 1, len(rewards)), avg_rewards, label=f'Moving Average (window size={window_size})')

    # Setting plot labels and titles
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def moving_average(data, window_size):
    return [np.mean(data[i:i+window_size]) for i in range(len(data) - window_size + 1)]


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, 64, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float32),
            nn.Tanh(),
            nn.Linear(64, n_actions, dtype=torch.float32)
        )

    def forward(self, x):

        return self.fc(x)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):

        self.state = self.env.reset()

        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        self.env.render()
        if np.random.random() < epsilon:
            action = random.choice(self.env.action)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(
        states, copy=False), dtype=torch.float32).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False), dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


def train_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PongEnvironment()
    net = DQN(env.observation_space,
                       env.action_space).to(device)

    tgt_net = DQN(env.observation_space,
                            env.action_space).to(device)

    buffer = ExperienceBuffer(REPLAY_SIZE)

    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    total_rewards = []
    mean_rewards = []
    total_loss = []
    t = 0
    episode = 0
    ts_step = 0
    ts = time.time()
    best_m_reward = None

    while True:
        t += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      t / EPSILON_DECAY_LAST_STEP)
        reward = agent.play_step(net, epsilon, device=device)
       
        if reward is not None:
            total_rewards.append(reward)
            episode += 1
            speed = (t - ts_step) / (time.time() - ts)
            ts_step = t
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            mean_rewards.append(m_reward)
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                      t, len(total_rewards), m_reward, epsilon,
                      speed
                  ))
            if best_m_reward is None or best_m_reward < m_reward:
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d time steps!!!!!!!!" % t)
                break
            
            if len(total_rewards) == 800:
                print("Algorithm does not converge. Start again")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if t % SYNC_TARGET_STEPS == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        total_loss.append(loss_t.detach().numpy())
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
    torch.save(tgt_net.state_dict(), 'model_weights.pth')
    plot_learning_curve(total_rewards)



