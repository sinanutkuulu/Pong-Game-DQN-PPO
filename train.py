#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

import PPO
import DQN


def train():
    DQN.train_agent()
    PPO.train_agent()
