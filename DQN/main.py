import gym
import numpy as np
import argparse
import tensorflow as tf
from DQNAgent import DQNAgent


argparser = argparse.ArgumentParser(description="DQN")

# CONFIGURATIONS
argparser.add_argument('--train', dest='is_train', action='store_true')
argparser.add_argument('--test', dest='is_train', action='store_false')
argparser.set_defaults(is_train=True)
argparser.add_argument("--model_dir", default="./model/cartpole-dqn.h5", type=str, 
  help="load pretrained weights")
argparser.add_argument("--log_dir", default="./model", type=str, 
  help="logging directory for tensorboard")
argparser.add_argument("--save_interval", default=10, type=int, 
  help="interval of saving models, default: 10 episodes")
argparser.add_argument("--env", default="CartPole-v1", type=str, 
  help="OpenAI gym environment")

# TRAINING HYPERPARAMETERS
argparser.add_argument("--episodes", default=1000, type=int, 
  help="number of episodes")
argparser.add_argument("--batch_size", default=32, type=int, 
  help="batch size")
argparser.add_argument("--punishment", default=-10, type=int, 
  help="negative reward on failure")

# AGENT HYPERPARAMETERS
argparser.add_argument("--gamma", default=0.95, type=float, 
  help="discount factor")
argparser.add_argument("--epsilon", default=1.0, type=float, 
  help="exploration rate")
argparser.add_argument("--epsilon_min", default=0.01, type=float, 
  help="minimal exploration rate")
argparser.add_argument("--epsilon_decay", default=0.995, type=float, 
  help="decay of exploration rate")
argparser.add_argument("--learning_rate", default=0.001, type=float, 
  help="learning rate")

args = argparser.parse_args()
print(args.is_train)

# setting up gym environment
env = gym.make(args.env)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# init agent
agent = DQNAgent(state_size, action_size, args)
agent.load(args.model_dir)

# logging setttings
file_writer = tf.summary.create_file_writer(args.log_dir + 'metrics')
file_writer.set_as_default()

# play the game
done = False
for e in range(args.episodes):
    state = env.reset() 
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else args.punishment
        next_state = np.reshape(next_state, [1, state_size])
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, args.episodes, time, agent.epsilon))
            tf.summary.scalar('score', data=time, step=e)
            break
        if len(agent.memory) > args.batch_size:
            agent.replay(args.batch_size)
    if args.is_train and e % args.save_interval == 0:
        agent.save(args.model_dir)
