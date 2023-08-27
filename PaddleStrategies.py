#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

from abc import ABC, abstractmethod
from pygame.locals import *


class AbstractPaddleStrategy(ABC):
    def __init__(self):
        self.paddle = None
        self.ball = None
        self.env = None
        self.action_set = [-1, 0, 1]

    def set_ball(self, ball):
        self.ball = ball

    def set_paddle(self, paddle):
        self.paddle = paddle

    def set_env(self, env):
        self.env = env

    @abstractmethod
    def move(self, action=0):
        pass


class SimpleAIStrategy(AbstractPaddleStrategy):
    def move(self, action=0):
        if self.ball.y > self.paddle.y + self.paddle.h - 20:
            self.paddle.update_with_velocity(1)
        elif self.ball.y < self.paddle.y + self.paddle.h - 20:
            self.paddle.update_with_velocity(-1)
        else:
        	self.paddle.update_with_velocity(0)

class ReinforcementLearningStrategy(AbstractPaddleStrategy):
    def move(self, action=0):
        action = action - 1
        self.paddle.update_with_velocity(action)
