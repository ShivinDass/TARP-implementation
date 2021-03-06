import gym
from env_wrappers import DiscretizeActionsWrapper
from models import LSTM_Encoder, Encoder
import sprites_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import copy
import numpy as np
import gym.spaces as spaces
import torch as th
import random

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
        # print(obs.shape)
        with th.no_grad():
            latent = self.encoder.forward(th.as_tensor(obs).float()).squeeze()
            # print(latent.shape)
        return np.array(latent)


env = ExpandImgDimWrapper(gym.make('Sprites-v2'))
# env = gym.make('SpritesState-v2')

trained_policy = PPO.load("saved_models/best_model/ppo_imageenc_v2")

obs = env.reset()
plt.figure()
im = plt.imshow(env.render())

prev_f = None
total_latent_loss = 0
for i in range(1000):
    action, _state = trained_policy.predict(obs, deterministic=True)

    n_obs, reward, done, info = env.step(action)
    
    with th.no_grad():
        print(trained_policy.policy.forward(th.tensor(obs).unsqueeze(dim=0), deterministic=True))
    print(action, reward)
    im.set_data(env.render())
    plt.show(block=False)

    obs = copy.deepcopy(n_obs)
    if done:
        print("done")
        obs = env.reset()
        prev_f = None
    l = input()

print("Total Latent Loss:", total_latent_loss)