#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""


from abc import ABC, abstractmethod
from Paddle import Paddle
import numpy as np


class AbstractCollusionStrategy(ABC):

    def __init__(self, left_paddle: Paddle, right_paddle: Paddle):
        self.left_paddle = left_paddle
        self.right_paddle = right_paddle
        self.ball = None
        self.env = None

    def set_ball(self, ball):
        self.ball = ball

    def set_environment(self, env):
        self.env = env

    @abstractmethod
    def check_x_collusion(self):
        pass

    def check_y_position(self):
        if self.ball.y + self.ball.r > self.env.y:
            self.ball.alpha = 2 * np.pi - self.ball.alpha
            self.ball.y = self.env.y - self.ball.r
        elif self.ball.y - self.ball.r < 100:
            self.ball.alpha = 2 * np.pi - self.ball.alpha
            self.ball.y = 100 + self.ball.r

    def check_and_act(self):
        self.check_y_position()
        return self.check_x_collusion()

class PositionCollusionStrategy(AbstractCollusionStrategy):
    def check_x_collusion(self):
        b_r_x = self.ball.x + self.ball.r
        b_l_x = self.ball.x - self.ball.r
        b_y = self.ball.y
        l_x = self.left_paddle.w
        l_u_y = self.left_paddle.y
        l_l_y = self.left_paddle.y + self.left_paddle.h
        l_c_d = (self.ball.y - (self.left_paddle.y + self.left_paddle.h // 2)) / (self.left_paddle.h // 2)
        r_u_y = self.right_paddle.y
        r_l_y = self.right_paddle.y + self.right_paddle.h
        r_x = self.right_paddle.x
        r_c_d = (self.ball.y - (self.right_paddle.y + self.right_paddle.h // 2)) / (self.right_paddle.h // 2)
        if not (l_x <= b_l_x):
            if l_u_y <= b_y <= l_l_y:
                self.ball.alpha = np.pi - self.ball.alpha
                if self.ball.alpha < 0:
                    self.ball.alpha += 2 * np.pi
                self.ball.alpha += (np.pi / 2)
                self.ball.alpha = self.ball.alpha % (2 * np.pi)
                self.ball.alpha += l_c_d * self.ball.d_alpha
                self.ball.alpha = np.clip(self.ball.alpha, np.pi / 12, np.pi - np.pi / 12)
                self.ball.alpha -= (np.pi / 2)
                if self.ball.alpha < 0:
                    self.ball.alpha += 2 * np.pi
            else:
                return -1
        elif not (b_r_x <= r_x):
            if r_u_y <= b_y <= r_l_y:
                self.ball.alpha = np.pi - self.ball.alpha
                if self.ball.alpha < 0:
                    self.ball.alpha += 2 * np.pi
                self.ball.alpha -= r_c_d * self.ball.d_alpha
                self.ball.alpha = self.ball.alpha % (2 * np.pi)
                self.ball.alpha = np.clip(self.ball.alpha, np.pi / 2 + np.pi / 12, 1.5 * np.pi - np.pi / 12)
            else:
                return 1
        return 0