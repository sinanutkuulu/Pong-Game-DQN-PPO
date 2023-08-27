#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sinan Utku Ulu

"""

import numpy as np

# General pygame parametersRLEnvironment
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
POINTS_TO_WIN = 10

# Paddle 1 Parameters
PADDLE_1_WIDTH = 10
PADDLE_1_HEIGHT = 150
PADDLE_1_V = 10
PADDLE_1_X = 0
PADDLE_1_Y = int(WINDOW_HEIGHT / 2 - PADDLE_1_HEIGHT / 2)
PADDLE_1_COLOR = (255, 255, 255)
PLAYER_1_IMAGE = "player_1.png"

# Paddle 2 Parameters
PADDLE_2_WIDTH = 10
PADDLE_2_HEIGHT = 200
PADDLE_2_V = 10
PADDLE_2_X = int(WINDOW_WIDTH - PADDLE_2_WIDTH)
PADDLE_2_Y = int(WINDOW_HEIGHT / 2 - PADDLE_1_HEIGHT / 2)
PADDLE_2_COLOR = (255, 255, 255)
PLAYER_2_IMAGE = "player_2.png"

# Ball Parameters
X_BALL = int(WINDOW_WIDTH / 2)
Y_BALL = int(WINDOW_HEIGHT / 2)
R_BALL = 25
V_X_BALL = 10
V_Y_BALL = 10
V_BALL = 15
ALPHA_BALL = 0.5
DELTA_ALPHA = np.pi / 6
BALL_COLOR = (255, 255, 255)

