import gym
import cv2
import random as rand

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


env_id = "ALE/Breakout-v5"
seed = 42
# env.reset(seed=seed)

env = make_vec_env(env_id, n_envs=1, seed=seed)
# env = gym.make(env_id, render_mode="human")
# env.reset(seed=seed)

##Load and train the model
model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000, progress_bar=True)
model.save("a2c_cartpole")

# del model

# model = A2C.load("a2c_cartpole")

###Get the evaluations
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
###Run the model

obs = env.reset()
for idx in range(1000):
   action, _states = model.predict(obs)
   obs, rewards, dones, info = env.step(action)
   env.render()
   if dones:
        obs = env.reset()
