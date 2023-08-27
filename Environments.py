#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

from CollusionStrategies import *
import pygame
import os
from Paddle import Paddle
from PaddleStrategies import *
from Ball import Ball
import parameters
import numpy as np


class PongEnvironment():
    def __init__(self, drawable=True):
        self.drawable = drawable
        self.episode = 0
        self.left_point = 0
        self.right_point = 0
        self.x = parameters.WINDOW_WIDTH
        self.y = parameters.WINDOW_HEIGHT
        self.window = pygame.display.set_mode((parameters.WINDOW_WIDTH, parameters.WINDOW_HEIGHT))
        paddle_l_strategy = SimpleAIStrategy()
        self.left_paddle = Paddle(parameters.PADDLE_1_X, parameters.PADDLE_1_Y, parameters.PADDLE_1_WIDTH,
                      parameters.PADDLE_1_HEIGHT,
                      parameters.PADDLE_1_V, parameters.PADDLE_1_COLOR, paddle_l_strategy, self.window,
                      image_name=parameters.PLAYER_1_IMAGE, paddle_type="L")
        paddle_r_strategy = ReinforcementLearningStrategy()
        self.right_paddle = Paddle(parameters.PADDLE_2_X, parameters.PADDLE_2_Y, parameters.PADDLE_2_WIDTH,
                      parameters.PADDLE_2_HEIGHT,
                      parameters.PADDLE_2_V, parameters.PADDLE_2_COLOR, paddle_r_strategy, self.window,
                      image_name=parameters.PLAYER_2_IMAGE, paddle_type="R")
        self.collusion_strategy = PositionCollusionStrategy(self.left_paddle, self.right_paddle)
        self.ball = Ball(self.collusion_strategy, self.window)
        paddle_l_strategy.set_ball(self.ball)
        paddle_r_strategy.set_ball(self.ball)
        self.collusion_strategy.set_environment(self)
        self.paddles = [self.left_paddle, self.right_paddle]
        for p in self.paddles:
            p.strategy.set_env(self)

        if drawable:
            pygame.init()
            self.window = pygame.display.set_mode((parameters.WINDOW_WIDTH, parameters.WINDOW_HEIGHT))
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 0)
            self.font_size = parameters.WINDOW_HEIGHT * 0.1
            self.font = pygame.font.SysFont("monospace", int(self.font_size))
            self.surface_point = self.font.render(str(self.left_point) + " - " + str(self.right_point), False,
                                                  parameters.TEXT_COLOR)
            self.surface_point_area = self.surface_point.get_rect()
            self.surface_point_area.center = (parameters.WINDOW_WIDTH / 2, 50)
        else:
            pygame.quit()
        self.done = False
        self.action_space = 3
        self.observation_space = 6
        self.action = [0, 1, 2]
        self.right_point_prev = 0
        self.left_point_prev = 0

    def render(self):
        if self.drawable:
            self.window.fill(parameters.BACKGROUND_COLOR)
            tx = self.font.render(str(self.left_point) + " - " + str(self.right_point), False,
                                  parameters.TEXT_COLOR)
            self.window.blit(tx, self.surface_point_area)
            self.ball.draw()
            for e in self.paddles:
                e.draw()
            pygame.draw.line(self.window, parameters.TEXT_COLOR, (0, 100), (parameters.WINDOW_WIDTH, 100), 2)
            pygame.display.update()
        else:
            print("Drawable was set to false so you cannot draw the environment!")

    def reset(self):
        self.done = False
        self.ball.reset()
        for e in self.paddles:
            e.reset()
        return self.observe()

    def move(self, action):
        self.ball.move()
        for e in self.paddles:
            e.move(action)

    def step(self, action):
        if not self.done:
            prev_observed = self.observe()
            self.move(action)
            res = self.collusion_strategy.check_and_act()
            self.right_point_prev = self.right_point
            self.left_point_prev = self.left_point
            self.update_point(res)
            obs_prime = self.observe()
            rew = self.get_reward(action, prev_observed, obs_prime, res)
            obs = np.array(obs_prime)
            if res != 0:
                self.done = True
            return obs, rew, self.done
        else:
            print("You are trying send an action to a finished episode!")
            exit(-1)

    def update_point(self, parameter):
        if parameter == -1:
            self.right_point += 1
        elif parameter == 1:
            self.left_point += 1

    def observe(self):
        # Get the positions of the paddles
        paddle_left_y = self.left_paddle.y
        paddle_right_y = self.right_paddle.y
        # Get the position and velocity of the ball
        ball_alpha = self.ball.alpha
        ball_x = self.ball.x
        ball_y = self.ball.y
        ball_x_v = self.ball.V * np.cos(ball_alpha)
        ball_y_v = self.ball.V * np.sin(ball_alpha)

        # Return a list as the game state
        state = [paddle_left_y, paddle_right_y,ball_x, ball_y,
                 ball_x_v, ball_y_v]
        return state

    def get_reward(self, action, prev_state, next_state, done):
        paddle_left_y_prev, paddle_right_y_prev, ball_x_prev, ball_y_prev, ball_v_x_prev, \
            ball_v_y_prev = prev_state
        paddle_left_y, paddle_right_y, ball_x, ball_y, ball_v_x, ball_v_y = next_state

        # Define rewards and penalties
        PERFECT_COLLUSION_REWARD = 0.1
        UP_DOWN_COLLUSION_PENALTY = -0.1
        WIN_REWARD = 10.0
        LOSS_PENALTY = -10.0
        TOWARDS_BALL_REWARD = 0.5
        AWAY_FROM_BALL_PENALTY = -0.5
        # Initial reward
        reward = 0.0
        # Check whether agent wins or loses the current episode
        if done:
            self.episode += 1
            if self.right_point > self.right_point_prev:
                reward += WIN_REWARD

            elif self.left_point >= self.left_point_prev:
                reward += LOSS_PENALTY

        else:

            # Check if ball has collided with the right paddle from the center
            if not ball_x + self.ball.r <= self.right_paddle.x:
               
                paddle_center = paddle_right_y + self.right_paddle.h / 2
                if paddle_center + 50 >= ball_y and paddle_center - 50 <= ball_y:
                    reward += PERFECT_COLLUSION_REWARD
                else:
                    reward += UP_DOWN_COLLUSION_PENALTY
            else:
                # Calculate the difference in the vertical distance between the ball and the paddle
                previous_distance = abs(prev_state[1] + self.right_paddle.h / 2 - prev_state[3])
                current_distance = abs(next_state[1] + self.right_paddle.h / 2 - next_state[3])

                # Check if the agent moved towards the ball
                if current_distance < previous_distance:

                            reward += TOWARDS_BALL_REWARD
                else:

                        reward += AWAY_FROM_BALL_PENALTY

        return reward



