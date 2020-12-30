import gym
import numpy as np
import argparse
import utils
import os.path
import tensorflow as tf
from DQNAgent import DQNAgent
from DDQNAgent import DDQNAgent


argparser = argparse.ArgumentParser(description="DQN")

# CONFIGURATIONS
argparser.add_argument('--train', dest='is_train', action='store_true')
argparser.add_argument('--test', dest='is_train', action='store_false')
argparser.set_defaults(is_train=True)

argparser.add_argument("--model", default="./model/cartpole-dqn.h5", type=str, 
  help="load pretrained weights")
argparser.add_argument("--log_file", default="./model/log.out", type=str, 
  help="log file")

argparser.add_argument("--save_interval", default=10, type=int, 
  help="interval of saving models, default: 10 episodes")
argparser.add_argument("--env", default="CartPole-v1", type=str, 
  help="OpenAI gym environment")

# TRAINING HYPERPARAMETERS
argparser.add_argument("--episodes", default=1000, type=int, 
  help="number of episodes")
argparser.add_argument("--punishment", default=-10, type=int, 
  help="negative reward on failure")
argparser.add_argument("--learning_rate", default=0.001, type=float, 
  help="learning rate")

args = argparser.parse_args()

# setting up gym environment
env = gym.make(args.env)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# init agent, load weights
agent = DDQNAgent(state_size, action_size, args)
if os.path.exists(args.model):
  agent.load(args.model)

utils.log(args.log_file, 'step,score')

# play the game
done = False
for episode in range(args.episodes):
    state = env.reset() 
    state = np.reshape(state, [1, state_size])
    for step in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else args.punishment
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            utils.log(args.log_file, str(episode) + ',' + str(step))
            break
        if len(agent.memory) > args.batch_size:
            agent.replay()
    if args.is_train and episode % args.save_interval == 0:
        agent.save(args.model)

