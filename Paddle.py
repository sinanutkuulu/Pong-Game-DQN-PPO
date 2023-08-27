#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""


from PaddleStrategies import *
import pygame
import parameters


class Paddle:
    def __init__(self, x, y, w, h, v, paddle_color, strategy: AbstractPaddleStrategy, window, image_name="avatar.png",
                 paddle_type="L"):
        self.x = x
        self.initial_y = y
        self.y = y
        self.w = w
        self.h = h
        self.v_list = [0]
        self.pos_list = []
        self.v = v
        self.strategy = strategy
        self.window = window
        self.paddle_type = paddle_type
        self.color = paddle_color
        if window is not None:
            self.image = pygame.image.load(image_name).convert()
            self.image = pygame.transform.scale(self.image, (100, 100))

        strategy.set_paddle(self)

    def update_position(self, new_y):
        v = new_y - self.y
        self.y = new_y
        self.v_list.append(v)
        self.pos_list.append(self.y)

    def update_with_velocity(self, direction):
        new_y = self.y + self.v * direction
        self.update_position(new_y)

    def reset(self):
        self.y = self.initial_y
        self.v_list = [0]
        self.pos_list = [self.y]

    def take_action(self):
        self.strategy.move()

    def move(self, action):
        self.strategy.move(action)
        if self.y < 100:
            self.y = 100
        if self.y > parameters.WINDOW_HEIGHT - self.h:
            self.y = parameters.WINDOW_HEIGHT - self.h

    def draw(self):
        pygame.draw.rect(self.window, self.color, (self.x, self.y, self.w, self.h))
        if self.paddle_type == "L":
            self.window.blit(self.image, (0, 0))
        else:
            self.window.blit(self.image, (parameters.WINDOW_WIDTH - 100, 0))
