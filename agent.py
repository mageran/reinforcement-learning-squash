import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras import layers, saving

#from tensorflow.keras.saving import register_keras_serializable

DATADIR = os.getenv('DATADIR', 'data')


def create_q_network(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

class DQNAgent:
    def __init__(self, state_size, action_size, filebasename='agent', load_filebasename=None):
        self.state_size = state_size
        self.action_size = action_size
        self.filebasename = filebasename
        self.load_filebasename = load_filebasename
        self.q_model = create_q_network(state_size, action_size)
        self.target_model = create_q_network(state_size, action_size)
        self.target_model.set_weights(self.q_model.get_weights())
        
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Exploration
        q_values = self.q_model.predict(state)
        return np.argmax(q_values[0])  # Exploitation
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = self.q_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q_values = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(next_q_values[0])
            states.append(state[0])
            targets.append(target[0])
        
        self.q_model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.q_model.get_weights())
    
    def check_no_overwrite(self):
        filepath = os.path.join(DATADIR, self.filebasename)
        fnames = [f"{filepath}_q_model.keras", f"{filepath}_target_model.keras"]
        for fname in fnames:
            if os.path.exists(fname):
                raise Exception(f"{fname} already exists; remove the file if you want to overwrite it or specify another name uising --save option")


    def save(self):
        os.makedirs(DATADIR, exist_ok=True)
        filepath = os.path.join(DATADIR, self.filebasename)
        print(f"saving models to {filepath}*.keras...")
        self.q_model.save(filepath + '_q_model.keras')
        self.target_model.save(filepath + '_target_model.keras')
    
    def load(self, basename=None):
        filepath = os.path.join(DATADIR, self.filebasename if basename is None else basename)
        print(f"loading models from {filepath}*.keras...")
        self.q_model = tf.keras.models.load_model(filepath + '_q_model.keras')
        self.target_model = tf.keras.models.load_model(filepath + '_target_model.keras')

