from skimage.transform import resize
import random
import numpy as np
import cv2

from DQN import *
from ExperienceReplayBuffer import *

class CatchEnv():
    def __init__(self):
        self.size = 21
        self.image = np.zeros((self.size, self.size))
        self.state = []
        self.fps = 4
        self.output_shape = (84, 84)

    def reset_random(self):
        self.image.fill(0)
        self.pos = np.random.randint(2, self.size-2)
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(self.size), 4
        self.image[self.bally, self.ballx] = 1
        self.image[-5, self.pos - 2:self.pos + 3] = np.ones(5)

        return self.step(2)[0]


    def step(self, action):
        def left():
            if self.pos > 3:
                self.pos -= 2
        def right():
            if self.pos < 17:
                self.pos += 2
        def noop():
            pass
        {0: left, 1: right, 2: noop}[action]()

        
        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size-1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        self.image[self.bally, self.ballx] = 1

        self.image[-5].fill(0)
        self.image[-5, self.pos-2:self.pos+3] = np.ones(5)
    
        terminal = self.bally == self.size - 1 - 4
        reward = int(self.pos - 2 <= self.ballx <= self.pos + 2) if terminal else 0

        [self.state.append(resize(self.image, (84, 84))) for _ in range(self.fps - len(self.state) + 1)]
        self.state = self.state[-self.fps:]

        return np.transpose(self.state, [1, 2, 0]), reward, terminal

    def get_num_actions(self):
        return 3

    def reset(self):
        return self.reset_random()

    def state_shape(self):
        return (self.fps,) + self.output_shape
    
    def reduce_dimensionality(self, state):
        grayscaled_state = cv2.cvtColor(np.float32(state), cv2.COLOR_BGR2GRAY)
        print("Grayscale state shape: {}".format(grayscaled_state.shape))
        return grayscaled_state



# -------------------------------------------------------------

def train_model(number_of_episodes, update_freq):
    # Initialize environment, RL-agent and memory buffer
    env = CatchEnv()
    agent = DeepQLearning()
    buffer = ExperienceReplayBuffer()

    # Let agent interact with environment, saving trajectories and learning
    for ep in range(number_of_episodes):
        run_environment(env, agent, buffer)
        batch = buffer.get_train_batch()
        if batch == None:
            continue
        loss = agent.train(batch)
        if ep % update_freq == 0:
            agent.update_target()
        print("End of the episode")


def run_environment(env, agent, buffer):
    state = env.reset()
    # cv2.imshow("image", state)
    # cv2.waitKey(0)

    state0, reward, terminal = env.step(1)

    while not terminal:
        action = agent.policy(state)
        state1, reward, terminal = env.step(action)
        buffer.save_trajectory(state0, action, reward, state1, terminal)

        state0 = state1

        # state, reward, terminal = env.step(random.randint(0,2))
        # print("Reward obtained by the agent: {}".format(reward))
        # state = np.squeeze(state)

        # #Reduce the dimensions of the state for the neural network
        # state = env.reduce_dimensionality(state)

        # #TODO feed the state to the neural network

        # #Resize the state to show it in a window
        
        # resized_state = resize(state, (254, 254))            
        # cv2.imshow("image", resized_state)
        # print("State shape: {}".format(resized_state.shape))
        # cv2.waitKey(0)

            


            

if __name__ == "__main__":
    train_model(100, 2)
    
    # env = CatchEnv()
    # env.reset()
    # state, reward, terminal = env.step(1)
    # state = np.squeeze(state)
    # state = env.reduce_dimensionality(state)

