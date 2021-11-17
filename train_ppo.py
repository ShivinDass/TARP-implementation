from stable_baselines3 import PPO
from stable_baselines3.common import policies
import numpy as np
import gym
from gym import spaces
from torch.nn.modules import activation
import sprites_env
from stable_baselines3.common.monitor import Monitor
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
from models import *
import models
from env_wrappers import *

# log_dir = "logs/cur/spritesv2_state"
# env = gym.make('SpritesState-v2')
# env = Monitor(env, log_dir)
# # eval_env = gym.make('SpritesState-v2')
# # eval_callback = EvalCallback(eval_env, best_model_save_path='saved_models/best_model/',
# #                              log_path='logs/best_model/', eval_freq=1200,
# #                              deterministic=True, render=False)
# model = PPO(policies.ActorCriticPolicy, env, verbose=1, batch_size=120, n_steps=240)#PPO('MlpPolicy', env, verbose=1)policies.ActorCriticPolicy
# model.learn(total_timesteps= int(1e6))#, callback=eval_callback)
# model.save("saved_models/best_model/ppo_state_v2")

# exit(0)

log_dir = "logs/cur/spritesv2_imageencoderfrozen"
env = gym.make('Sprites-v2')

with_encoder_wrapper = False
env = EncoderWrapper(env, "saved_models/encoder_model") if with_encoder_wrapper else ExpandImgDimWrapper(env)
env = Monitor(env, log_dir)

# eval_env = gym.make('Sprites-v0')
# eval_env = EncoderWrapper(eval_env, "saved_models/encoder_model") if with_encoder_wrapper else ExpandImgDimWrapper(eval_env)
# eval_callback = EvalCallback(eval_env, best_model_save_path='saved_models/best_model/',
#                              log_path='logs/best_model/', eval_freq=20000,
#                              deterministic=True, render=False)


if with_encoder_wrapper:
    model = PPO(policies.ActorCriticPolicy, env, clip_range=0.2, n_epochs=10, batch_size=120, n_steps=240, verbose=1)
else:
    policy_kwargs = dict(
        features_extractor_class = CustomFeatureExtractor, normalize_images=False#, net_arch = [{'pi':[32,32], 'vf':[32,32]}]
    )
    model = PPO(policies.ActorCriticPolicy, env, clip_range=0.1, learning_rate=1e-5, n_epochs=10, batch_size=120, n_steps=240, verbose=1, policy_kwargs = policy_kwargs)
    checkpoint = th.load("saved_models/autoencoder_encoder")
    model.policy.features_extractor.encoder.load_state_dict(checkpoint)
    
model.learn(total_timesteps=int(1e6))#, callback=eval_callback)
model.save("saved_models/best_model/ppo_imageencoderfrozen_v2")