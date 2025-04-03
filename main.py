import os
import sys
import time
import numpy as np
from env import SquashEnv
from agent import DQNAgent
from utils import *
from argparse import ArgumentParser

def train(episodes=1, do_render=False, filebasename='agent', load_filebasename=None):
    env = SquashEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, filebasename=filebasename)
    if filebasename != load_filebasename:
        agent.check_no_overwrite()
    if load_filebasename is not None:
        agent.load(basename=load_filebasename)
    #episodes = 100
    for e in range(episodes):
        orig_stdout = sys.stdout
        try:
            print_red_yellow_bold(f"training episode {e} start...")
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            done = False
            env.episode_direction = 0
            while not done:
                action = agent.act(state)
                if env.episode_direction != 0:
                    env.episode_direction = action
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

            print_blue("replaying...", end="", flush=True)
            sys.stdout = open(os.devnull, "w")
            agent.replay()
            agent.update_target_model()
            sys.stdout = orig_stdout
            print_blue("done.")
            print(f"training episode {e} end")
        except KeyboardInterrupt:
            sys.stdout = orig_stdout
            print(f"Keyboard interrupt during episode {e+1}")
            break
        finally:
            sys.stdout = orig_stdout

    
    agent.save()

def evaluate_agent(episodes=5, filebasename='agent'):
    env = SquashEnv(is_training=False, do_render=True)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, filebasename=filebasename)
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
    episodes = args.episodes
    filebasename = args.save
    load_filebasename = args.load
    update_filebasename = args.update
    if update_filebasename is not None:
        load_filebasename = update_filebasename
        filebasename = update_filebasename
    train(episodes, filebasename=filebasename, load_filebasename=load_filebasename)

def command_eval(args):
    print(args)
    episodes = args.episodes
    filebasename = args.load
    evaluate_agent(episodes, filebasename=filebasename)

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="commands")
    train_parser = subparsers.add_parser("train", help="run training")
    train_parser.add_argument("-e", "--episodes", type=int, default=10, help="number of episodes for training")
    train_parser.add_argument("-s", "--save", type=str, default='agent', help="basename of the model file")
    train_parser.add_argument("-l", "--load", type=str, default=None, help="initialize the model with a previously created one with this basename")
    train_parser.add_argument("-u", "--update", type=str, default=None, help="basename for the model to update")
    train_parser.set_defaults(func=command_train)

    eval_parser = subparsers.add_parser("eval", help="run evaluation")
    eval_parser.add_argument("-e", "--episodes", type=int, default=10, help="number of episodes for evaluation")
    eval_parser.add_argument("-l", "--load", type=str, default='agent', help="basename of the model file (previously created during training)")
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
