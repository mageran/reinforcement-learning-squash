import gym
import gym.spaces
import time
import numpy as np
from visualize import SquashVisualizer
from utils import *

PADDLE_DIFF = 0.125

class SquashEnv(gym.Env):
    def __init__(self, is_training=True, randomize_initial_state=True, do_render=False):
        super(SquashEnv, self).__init__()
        self.is_training = is_training
        self.randomize_initial_state = randomize_initial_state
        self.do_render = do_render
        # action space (left, right, nothing)
        self.action_space = gym.spaces.Discrete(3)
        # observation space (ball position (x and y), velocity (dx, dy), paddle position (x))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.visualizer = SquashVisualizer(self) if do_render else None
        self.reset()


    
    def reset(self):
        if self.randomize_initial_state:
            # Random ball position (near the center, but with slight variation)
            self.ball_position = np.array([np.random.uniform(0.2, 0.8), np.random.uniform(0, 0.1)])

            # Random ball velocity (small values for gradual movement)
            self.ball_velocity = np.array([np.random.choice([-0.01, 0.01]) * np.random.uniform(0.5, 1.5),
                                        np.random.uniform(0.01, 0.03)])

            # Random paddle position (somewhere near the center)
            self.paddle_position = np.random.uniform(0.4, 0.6)
        else:
            self.ball_position = np.array([0.5, 0])
            self.ball_velocity = np.array([0.01, 0.01])
            self.paddle_position = 0.5

        self.ball_target_x = None
        if self.is_training:
            self.simulate_move_ball_to_bottom(render=False)
        self.done = False

        # Return initial state
        return np.concatenate([self.ball_position, self.ball_velocity, [self.paddle_position]])
    
    def move_ball(self):
        # Update ball position based on its velocity
        self.ball_position += self.ball_velocity
        # Ball collision with the walls
        if self.ball_position[0] <= 0 or self.ball_position[0] >= 1:
            self.ball_velocity[0] = -self.ball_velocity[0]
        if self.ball_position[1] <= 0 or self.ball_position[1] >= 1:
            self.ball_velocity[1] = -self.ball_velocity[1]

    def simulate_move_ball_to_bottom(self, render=False, delay=0.3):
        print_green(f"simulation starting position: {self.ball_position}, velocity: {self.ball_velocity}")
        saved_ball_position = np.copy(self.ball_position)
        saved_ball_velocity = np.copy(self.ball_velocity)
        ball_move_counter = 0
        def _render(delay=delay):
            if self.do_render and render:
                self.visualizer.set_message(f"Ball at {self.ball_position}")
                self.render()
                time.sleep(delay)
        while self.ball_position[1] < 1:
            self.move_ball()
            ball_move_counter += 1
        # store the x position of the ball when it hits the bottom
        # as this will be the target for the paddle to go
        self.ball_target_x = self.ball_position[0]
        if self.do_render and render:
            _render()
        print(f"ball hits the bottom at x={self.ball_position[0]}, ball moved {ball_move_counter} times")
        self.ball_position = saved_ball_position
        self.ball_velocity = saved_ball_velocity

    @property
    def ball_target_paddle_distance(self):
        distance = abs(self.ball_target_x - self.paddle_position)
        at_target = distance < PADDLE_DIFF
        return distance, at_target

    def move_paddle(self, action):
        if action == 0:  # Move left
            self.paddle_position -= 0.05
            print_blue("paddle moved to the left")
        elif action == 1:  # Move right
            self.paddle_position += 0.05
            print_blue("paddle moved to the right")
        else:
            print_blue("paddle did not move")

    def training_step(self, action):
        distance_before, was_at_target = self.ball_target_paddle_distance
        self.move_paddle(action)
        distance_after, is_at_target = self.ball_target_paddle_distance
        if was_at_target:
            reward = 1 if is_at_target else -1
        else:
            if is_at_target:
                reward = 1
            else:
                reward = 1 if distance_after < distance_before else -1
        self.move_ball()
        self.done = self.ball_position[1] >= 1
        str = "got shorter" if distance_after < distance_before else "got longer" if distance_before < distance_after else "stayed the same"
        print(f"=> distance from paddle to ball target position {str}")
        print(f"=> was at target: {was_at_target}, is_at_target: {is_at_target}")
        print_green_grey(f"=> reward: {reward}")
        return np.concatenate([np.copy(self.ball_position), np.copy(self.ball_velocity), [self.paddle_position]]), reward, self.done, {}
 
    def reset_if_ball_at_bottom(self):
        if self.ball_position[0] >= 1:
            self.reset()

    def step(self, action):
        if self.is_training:
            return self.training_step(action)
        else:
            return self.eval_step(action)

    def eval_step(self, action):
        self.move_paddle(action)
        # Reward: The agent gets a positive reward for hitting the ball and a negative reward for missing.
        self.move_ball()
        reward = 0
        if self.ball_position[1] >= 1:
            print(f"ball hits bottom at {self.ball_position[0]}")
            print(f"simulated position: {self.ball_target_x}")
            if abs(self.ball_position[0] - self.paddle_position) < PADDLE_DIFF:
                reward = 1  # Hit the ball
                self.visualizer.set_message("HIT THE BALL")
                #self.done = True
            #elif self.ball_position[1] < 0:
            else:
                reward = -1  # Miss the ball
                self.done = True  # End the episode
                self.visualizer.set_message("MISSED")
        # Return the updated state and reward
        return np.concatenate([self.ball_position, self.ball_velocity, [self.paddle_position]]), reward, self.done, {}
    
    def render(self, delay=16):
        if self.do_render:
            self.visualizer.render(delay=delay)
        