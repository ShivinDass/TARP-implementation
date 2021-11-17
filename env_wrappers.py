import gym.spaces as spaces
import numpy as np
import gym
from models import *

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
    Changes observation image dim from (dim,dim) to (1,dim,dim) and converts pixels from 0-1 in 0-255
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
            latent = self.encoder(th.as_tensor(obs).float()).squeeze()
        return np.array(latent)

class DiscretizeActionsWrapper(gym.ActionWrapper):
    mapping = {
        0:  (1,0),
        1:  (1,1),
        2:  (0,1),
        3:  (-1,1),
        4:  (-1,0),
        5:  (-1,-1),
        6:  (0,-1),
        7:  (1,-1)
    }
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(8)
    
    def action(self, action):
        return self.mapping[action]