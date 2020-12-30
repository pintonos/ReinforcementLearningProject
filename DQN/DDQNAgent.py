import random
import numpy as np
from collections import deque
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# set random seed
random.seed(123456)


class DDQNAgent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size

        # agent hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # min. exploration rate
        self.epsilon_decay = 0.995
        self.learning_rate = args.learning_rate
        self.num_units = [100, 100]  # number of units in each layer (except output layer)
        self.batch_size = 32
        
        self.target_update_interval = 100
        self.update_count = 0

        self.train = args.is_train
        if self.train:
            self.epsilon = 0.0

        self.model = self._build_model()
        # second network
        self.target_model = self._build_model()


    def _build_model(self):
        model = Sequential()
        # input layer
        model.add(Dense(self.num_units[0], input_dim=self.state_size, activation='relu'))
        # hidden layers
        for hidden_units in self.num_units[1:]:     
            model.add(Dense(hidden_units, activation='relu'))
        # output layer
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights)


    def memorize(self, state, action, reward, next_state, done):
        if not self.train: # dont memorize in testing mode
            return
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self):
        # maybe don't resample on every timestep (see readme)
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state[0])
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target model 
        self.update_count += 1
        if self.update_count > self.target_update_interval:
            self.update_target_model()
            self.update_count = 0


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)