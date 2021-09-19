from stable_baselines3 import PPO
from stable_baselines3.common import policies
import numpy as np
import gym

import sprites_env

venv = gym.make('SpritesState-v1')
model = PPO('MlpPolicy', venv, verbose=1, batch_size=100, n_steps=200)#PPO('MlpPolicy', env, verbose=1)policies.ActorCriticPolicy
model.learn(total_timesteps= int(5e5))
model.save("saved models/ppo_state_policy_distractor1")


venv = gym.make('SpritesState-v2')
model = PPO('MlpPolicy', venv, verbose=1, batch_size=100, n_steps=200)#PPO('MlpPolicy', env, verbose=1)policies.ActorCriticPolicy
model.learn(total_timesteps= int(5e5))
model.save("saved models/ppo_state_policy_distractor2")