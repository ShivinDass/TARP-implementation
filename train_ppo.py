from stable_baselines3 import PPO
from stable_baselines3.common import policies
import numpy as np
import gym
from gym import spaces
import sprites_env
from models import CustomFeatureExtractor


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


env = gym.make('Sprites-v0')
env = ExpandImgDimWrapper(env)
print(env.observation_space)
print(env.reset().shape)

policy_kwargs = dict(
    features_extractor_class = CustomFeatureExtractor
)
model = PPO(policies.ActorCriticCnnPolicy, env, verbose=1, batch_size=100, n_steps=200, policy_kwargs = policy_kwargs)
print(type(model.policy.features_extractor))
model.learn(total_timesteps=int(5e5))