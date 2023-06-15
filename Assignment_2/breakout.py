import gym
import cv2
import random as rand
import time

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed


env_id = "ALE/Breakout-v5"
seed = 42
# env.reset(seed=seed)

env = make_vec_env(env_id, n_envs=1, seed=seed)
print('test')
env.render(mode = "human")
env = VecFrameStack(env, n_stack=4)
# env = gym.make(env_id, render_mode="human")
# env.reset(seed=seed)

##Load and train the model
model = A2C("CnnPolicy", env, verbose=1, n_steps=50, learning_rate = 0.005, gamma = 0.99, gae_lambda = 0.95, ent_coef = 0.01, vf_coef = 0.25, max_grad_norm = 0.5, rms_prop_eps = 1e-5)
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000, progress_bar=True)
# model.save("/home/anne-jan/Documents/drl_dataset/a2c_cartpole")

# del model

model = A2C.load("/home/anne-jan/Documents/drl_dataset/a2c_breakout_25m", env=env)

###Get the evaluations
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
###Run the model

obs = env.reset()
for idx in range(1000):
   # time.sleep(0.05)
   time.sleep(0.25)
   action, _states = model.predict(obs)
   obs, rewards, done, info = env.step(action)
   env.render(mode = "human")
   if done:
        obs = env.reset()
