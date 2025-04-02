import os
import sys
import time
import numpy as np
from env import SquashEnv
from agent import DQNAgent
from utils import *
from argparse import ArgumentParser


def train(episodes=1, do_render=False):
    env = SquashEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #episodes = 100
    for e in range(episodes):
        print_red_yellow_bold(f"training episode {e} start...")
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if do_render:
                env.render()
            #if e % 1 == 0:
                #env.render(delay=1)

            if done:
                print_green_white_bold(f"Episode {e+1}/{episodes}, Reward: {total_reward}")
                break

        orig_stdout = sys.stdout
        print_blue("replaying...", end="", flush=True)
        sys.stdout = open(os.devnull, "w")
        agent.replay()
        agent.update_target_model()
        sys.stdout = orig_stdout
        print_blue("done.")
        print(f"training episode {e} end")
    
    agent.save()

def evaluate_agent(episodes=5):
    env = SquashEnv(is_training=False, do_render=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load()
    agent.epsilon = 0  # Disable exploration
    state_size = env.observation_space.shape[0]
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            total_reward += reward

            # Render the game
            env.render()

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    env.close()        

def command_train(args):
    #print(f"train args: {args}")
    episodes = args.episodes
    train(episodes)

def command_eval(args):
    episodes = args.episodes
    evaluate_agent(episodes)

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="commands")
    train_parser = subparsers.add_parser("train", help="run training")
    train_parser.add_argument("-e", "--episodes", type=int, default=10)
    train_parser.set_defaults(func=command_train)

    eval_parser = subparsers.add_parser("eval", help="run evaluation")
    eval_parser.add_argument("-e", "--episodes", type=int, default=10)
    eval_parser.set_defaults(func=command_eval)

    args = parser.parse_args()
    if hasattr(args, "func"):
        try:
            args.func(args)
        except Exception as e:
            print_red_yellow_bold(f"*** {e}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
