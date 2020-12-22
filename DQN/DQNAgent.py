import random
import numpy as np
from collections import deque
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# set random seed
random.seed(123456)

class DQNAgent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate

        self.train = args.is_train
        if self.train:
            self.epsilon = 0.0

        self.tb_callback = callbacks.TensorBoard(log_dir=args.log_dir)

        self.model = self._build_model()

    def _build_model(self):
        # TODO replace with a real "deep" network

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        if not self.train: # dont memorize in testing mode
            return
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # move this line outside for loop?
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[self.tb_callback])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
