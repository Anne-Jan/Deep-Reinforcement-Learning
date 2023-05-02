import gym
import cv2
import random as rand

# env = gym.make("ALE/Atlantis-v5", render_mode="human")
env = gym.make("ALE/Breakout-v5", render_mode="human")

env.action_space.seed(42)

observation, info = env.reset(seed=42)
prev_action = None
for idx in range(100):
    #Action 0: No action
    #Action 1: Fire button
    #Action 2: Fire left
    #Action 3: Fire right
    # if info["episode_frame_number"] < 200:
    if prev_action != 0: 
        random_action = 0
    else:
        random_action = rand.randint(1,3)
    prev_action = random_action
    # random_action = 1
    # random_action = rand.randint(1,2)
    # random_action = 2
    print(random_action)
    observation, reward, terminated, truncated, info = env.step(int(random_action))
    print("Info: {}, Reward: {}, Terminated: {}, Truncated: {}".format(info, reward, terminated, truncated))

    if terminated or truncated:
        observation, info = env.reset()

env.close()