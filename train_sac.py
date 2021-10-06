from stable_baselines3 import PPO, SAC
from stable_baselines3.common import policies
import numpy as np
import gym
from gym import spaces
import sprites_env
from models import CustomFeatureExtractor, Encoder
from stable_baselines3.common.monitor import Monitor
import torch as th
from stable_baselines3.common.callbacks import EvalCallback
import models


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

class ExpandImgDimWrapper2(gym.core.ObservationWrapper):
    """
    Changes observation image dim from (dim,dim) to (1,dim,dim)
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=0, high=255,
                shape=(1,env.resolution, env.resolution),
                dtype=np.uint8)
        self.resolution = env.resolution

    def observation(self, obs):
        return np.expand_dims((obs*255).astype(np.uint8),axis=0)

class EncoderWrapper(gym.core.ObservationWrapper):
    """
    Encodes Observation
    """

    def __init__(self, env, encoder_path):
        super().__init__(env)

        self.encoder = Encoder()
        self.encoder.load_state_dict(th.load(encoder_path))
        for params in self.encoder.parameters():
            params.requires_grad_ = False

        self.observation_space = spaces.Box(low = 0.0, high = 1.0,
                shape=(64,),
                dtype=np.float32
        )

    def observation(self, obs):
        obs = obs.reshape(1, 1,obs.shape[0], obs.shape[1])
        with th.no_grad():
            latent = self.encoder(th.as_tensor(obs).float()).squeeze()#.forward
            
        return np.array(latent)

log_dir = "logs/cur/spritesv2_state_sac"
env = gym.make('SpritesState-v0')
env = Monitor(env, log_dir)
# eval_env = gym.make('SpritesState-v2')
# eval_callback = EvalCallback(eval_env, best_model_save_path='saved_models/best_model/',
#                              log_path='logs/best_model/', eval_freq=1200,
#                              deterministic=True, render=False)
model = SAC('MlpPolicy', env, verbose=1)#, batch_size=100, n_steps=200)#PPO('MlpPolicy', env, verbose=1)policies.ActorCriticPolicy
model.learn(total_timesteps= int(1e5))#, callback=eval_callback)
# model.save("saved_models/ppo_state_policy_distractor1")

exit(0)

log_dir = "logs/cur/spritesv0_encoded"
env = gym.make('Sprites-v0')

with_encoder_wrapper = False
env = EncoderWrapper(env, "saved_models/encoder_model") if with_encoder_wrapper else ExpandImgDimWrapper(env)
# env = Monitor(env, log_dir)

eval_env = gym.make('Sprites-v0')
eval_env = EncoderWrapper(eval_env, "saved_models/encoder_model") if with_encoder_wrapper else ExpandImgDimWrapper(eval_env)
eval_callback = EvalCallback(eval_env, best_model_save_path='saved_models/best_model/',
                             log_path='logs/best_model/', eval_freq=1200,
                             deterministic=True, render=False)


if with_encoder_wrapper:
    model = PPO(policies.ActorCriticPolicy, env, clip_range=0.2, n_epochs=10, batch_size=120, n_steps=240, verbose=0)
else:
    policy_kwargs = dict(
        features_extractor_class = CustomFeatureExtractor, normalize_images=False
    )
    model = SAC(policies.SACPolicy, env, clip_range=0.2, n_epochs=10, batch_size=120, n_steps=240, verbose=1, policy_kwargs = policy_kwargs)
    checkpoint = th.load("saved_models/encoder_model")
    model.policy.features_extractor.encoder.load_state_dict(checkpoint)
    # for param in model.policy.features_extractor.parameters():
    #     param.requires_grad_ = False

model.learn(total_timesteps=int(2e5))#, callback=eval_callback)
model.save("saved_models/best_model/ppo_encoder_v0")