import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import RMSprop
# from collections import deque 
# from tensorflow import gather_nd
# from tensorflow.keras.losses import mean_squared_error 



def create_model():

    dqn_model = models.Sequential()
    dqn_model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (84, 84, 1)))
    dqn_model.add(layers.MaxPooling2D((2, 2)))
    dqn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    dqn_model.add(layers.MaxPooling2D((2, 2)))
    dqn_model.add(layers.Flatten())

    dqn_model.add(layers.Dense(64, activation='relu'))
    dqn_model.add(layers.Dense(3))

    dqn_model.compile(optimizer=tf.optimizers.AdamW(learning_rate=0.001), loss='mse')

    return dqn_model

class DeepQLearning:
    def __init__(self):
        self.dqn_model = create_model()
        self.target_dqn_model = create_model()

    # The policy takes a state and returns an action
    def policy(self, state):
        # Add small chance to explore
        if random.random() < 0.1:
            return random.randint(0, 2)
        state = state.reshape(1, 84, 84, 1)
        model_output = self.dqn_model.predict(state)
        # TODO: check of dit juiste format output heeft zo
        action = np.argmax(model_output)
        return action

    def train(self, batch):
        state0_batch, action_batch, reward_batch, state1_batch, terminal_batch = batch
        current_q = self.dqn_model(state0_batch)
        target_q = np.copy(current_q)
        next_q = self.dqn_model(state1_batch)
        best_next_q = np.amax(next_q, axis=1)
        for idx in range(state0_batch.shape[0]): # testen of length ook werkt
            if terminal_batch[idx]:
                target_q[idx][action_batch[idx]] = reward_batch[idx]
            else:
                target_q[idx][action_batch[idx]] = reward_batch[idx] + 0.95 * best_next_q[idx]

        result = self.dqn_model.fit(x=state0_batch, y=target_q)

        return result.history['loss']
    
    def update_target(self):
        self.target_dqn_model.set_weights(self.dqn_model.get_weights())
        
