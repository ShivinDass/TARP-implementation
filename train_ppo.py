from stable_baselines3 import PPO
from stable_baselines3.common import policies
import numpy as np
import gym
from gym import spaces
import sprites_env
from models import CustomFeatureExtractor
from stable_baselines3.common.monitor import Monitor
import torch as th
from stable_baselines3.common.callbacks import EvalCallback



class ExpandImgDimWrapper(gym.core.ObservationWrapper):
    """
    Changes observation image dim from (dim,dim) to (1,dim,dim)
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=0.0, high=1,
                shape=(1,env.resolution, env.resolution),
                dtype=np.float32)
        self.resolution = env.resolution

    def observation(self, obs):
        return np.expand_dims(obs,axis=0)


# venv = gym.make('SpritesState-v1')
# model = PPO('MlpPolicy', venv, verbose=1, batch_size=100, n_steps=200)#PPO('MlpPolicy', env, verbose=1)policies.ActorCriticPolicy
# model.learn(total_timesteps= int(5e5))
# model.save("saved models/ppo_state_policy_distractor1")

log_dir = "logs/cur/spritesv0_encoded"
env = gym.make('Sprites-v0')
env = ExpandImgDimWrapper(env)
env = Monitor(env, log_dir)
# print(env.observation_space)
# print(env.reset().shape)

eval_env = ExpandImgDimWrapper(gym.make('Sprites-v0'))
eval_callback = EvalCallback(eval_env, best_model_save_path='saved_models/best_model/',
                             log_path='logs/best_model/', eval_freq=1200,
                             deterministic=True, render=False)

policy_kwargs = dict(
    features_extractor_class = CustomFeatureExtractor
)
model = PPO(policies.ActorCriticCnnPolicy, env, verbose=1, batch_size=100, n_steps=200, policy_kwargs = policy_kwargs)
checkpoint = th.load("saved_models/encoder_model")
model.policy.features_extractor.encoder.load_state_dict(checkpoint)
# for param in model.policy.features_extractor.parameters():
#     param.requires_grad_ = False
# print(type(model.policy.features_extractor))
model.learn(total_timesteps=int(6e5), callback=eval_callback)
model.save("saved_models/ppo_encoder_v0")