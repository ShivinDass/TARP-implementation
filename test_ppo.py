import gym
from models import LSTM_Encoder
import sprites_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import copy
import numpy as np
import gym.spaces as spaces

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

# env = gym.make('SpritesState-v2')
env = ExpandImgDimWrapper(gym.make('Sprites-v0'))

trained_policy = PPO.load("saved_models/best_model/best_model")

obs = env.reset()
plt.figure()
im = plt.imshow(env.render())
while 1>0:
    action, _state = trained_policy.predict(obs, deterministic=True)
    n_obs, reward, done, info = env.step(action)
    
    print(action, reward)
    im.set_data(env.render())
    plt.show(block=False)

    obs = copy.deepcopy(n_obs)
    if done:
        print("done")
        obs = env.reset()
    l = input()