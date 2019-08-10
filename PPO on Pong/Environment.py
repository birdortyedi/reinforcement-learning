from CollusionStrategies import *
import os
import pickle as pkl
import numpy as np
from Paddle import Paddle
from PaddleStrategies import *
import matplotlib.pyplot as plt
import datetime
from Ball import Ball
import parameters
import pygame

"""
Structure is a bit complex, so a quick tip for you:

The class variables you CAN use for reward calculation and state definition:
-self.left_paddle
-self.right_paddle
-self.ball
-self.left_point
-self.right_point

All other variables are intended for internal structure. If you think any other parameter from the environment is needed,
contact your TA about it.
"""


class PongEnvironment:
    def __init__(self, drawable=True):
        self.drawable = drawable
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
        return np.array([self.left_paddle.x, self.right_paddle.x, self.left_paddle.y, self.right_paddle.y,
                        self.ball.x, self.ball.y, self.ball.alpha])

    def get_reward(self, action, prev_state, next_state, res):
        # if res is -1, RL agent wins, if res is 1, opponent wins, else, game is still going on
        if res == -1:
            return 500
        elif res == 1:
            return -100
        else:
            return - np.sqrt((self.right_paddle.x - self.ball.x)**2 + (self.right_paddle.y - self.ball.y)**2) / \
                   parameters.WINDOW_WIDTH
