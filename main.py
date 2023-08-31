#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

import train
import test
from Environments import PongEnvironment


if __name__ == '__main__':
    total_reward = 0.0
    env = PongEnvironment()
    train.train()
    test.test_DQN(env, total_reward)