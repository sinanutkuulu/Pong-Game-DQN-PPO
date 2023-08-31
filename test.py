#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

import time
from DQN import DQN
from PPO import Agent
import torch
import numpy as np
from Environments import PongEnvironment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_DQN = PongEnvironment()

total_win = 0
total_loss = 0
total_reward = 0.0


def test_DQN(env, total_reward):

    PATH_DQN = 'model_weights.pth'
    dqn_model = DQN(env.observation_space, env.action_space).to(device)
    dqn_model.eval()
    dqn_model.load_state_dict(torch.load(PATH_DQN, map_location=device))
    with torch.no_grad():
        for i in range(10):
            print("Test", str(i+1), "starting...")
            for episode in range(100):
                state = env.reset()
                while True:
                    env.render()
                    time.sleep(0.001)
                    state_v = torch.tensor(np.array([state], copy=False), dtype=torch.float32)
                    q_vals = dqn_model(state_v).data.numpy()[0]
                    action = np.argmax(q_vals)

                    state, reward, done = env.step(action)
                    total_reward += reward
                    if done:
        
                        break
            print("End of the 100 games")
            print("Left: " + str(env.left_point) + " Right: " + str(env.right_point))


env_PPO = PongEnvironment()


def test_PPO(env, total_reward):
    PATH_PPO = 'ppo_params.pth'
    ppo_model = Agent(n_actions=env.action_space,
                        input_dims=env.observation_space)
    actor = ppo_model.actor
    actor.eval()
    actor.load_state_dict(torch.load(PATH_PPO, map_location=device))

    with torch.no_grad():
        for episode in range(100):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                env.render()
                time.sleep(0.001)
                action, prob, val = ppo_model.choose_action(observation)
                observation_, reward, done = env.step(action)
                score += reward
            print('episode', episode+1, 'score %.1f' % score)

if __name__ == '__main__':
    pass
    #test_DQN(env_DQN, total_reward)
    #test_PPO(env_PPO, total_reward)