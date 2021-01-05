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

argparser.add_argument("--model", default="./model/ddqn-acrobot.h5", type=str, 
  help="load pretrained weights")
argparser.add_argument("--log_file", default="./model/ddqn-acrobot.out", type=str, 
  help="log file")

argparser.add_argument("--save_interval", default=50, type=int, 
  help="interval of saving models, default: 50 episodes")
argparser.add_argument("--env", default="Acrobot-v1", type=str, 
  help="OpenAI gym environment")

# TRAINING HYPERPARAMETERS
argparser.add_argument("--episodes", default=1000, type=int, 
  help="number of episodes")
argparser.add_argument("--max_steps", default=500, type=int, 
  help="max. number of steps per episode")
argparser.add_argument("--punishment", default=-1, type=int, 
  help="negative reward on failure")

args = argparser.parse_args()


def main():
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
  for episode in range(args.episodes):
      state = env.reset() 
      state = np.reshape(state, [1, state_size])

      done = False
      step = 0
      sum_reward = 0
      while not done and step < args.max_steps: # break up after max_steps or when done
          # env.render()
          action = agent.act(state)
          next_state, reward, done, _ = env.step(action)
          reward = reward if not done else args.punishment
          sum_reward += reward
          next_state = np.reshape(next_state, [1, state_size])

          # write into replay buffer
          agent.memorize(state, action, reward, next_state, done)
          state = next_state

          # update agent
          agent.replay()
          step += 1

      utils.log(args.log_file, str(episode) + ',' + str(sum_reward))

      if args.is_train and episode % args.save_interval == 0:
          agent.save(args.model)


if __name__ == '__main__':

  # CLUSTER SETUP #
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  for r in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[r], True)

  try:
    import cluster_setup
  except ImportError:
    pass
  
  main()
