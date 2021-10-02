import torch as th
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from models import *
import gym
from gym import spaces
import sprites_env
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class Checker(nn.Module):

    def __init__(self, state_dim):
        super(Checker, self).__init__()
        self.encoder = Encoder()

        self.decoder  = nn.Sequential(
            nn.Linear(in_features=64, out_features=32), nn.LeakyReLU(),
            nn.Linear(in_features=32,out_features=32), nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=state_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.encoder(input)
        # print(th.max(x, dim=1), th.min(x, dim=1))
        return self.decoder(x)

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



env = gym.make('SpritesState-v0')
obs = env.reset()

# env = ExpandImgDimWrapper(env)
model = Checker(len(obs))

checkpoint = th.load("saved_models/encoder_model")
model.encoder.load_state_dict(checkpoint)
# for params in model.encoder.parameters():
#     params.requires_grad_ = False

optimizer = optim.Adam(model.parameters(), lr = 0.0005)
loss = th.nn.MSELoss()

batch_size = 32
n_steps = 10
n_epoch = 1000

for i in range(n_epoch):
    total_loss = 0

    obs = env.reset()
    # labels.append(env.getState().reshape(-1))
    # train_data.append(obs)
    for _ in range(n_steps):
        train_data = []
        labels = []
        for _ in range(batch_size):
            obs, reward, done, info = env.step(env.action_space.sample())

            if done:
                obs = env.reset()

            train_data.append(env.render().reshape(1,64,64))
            labels.append(obs)

        
        train_data = th.as_tensor(train_data).float()
        labels = th.as_tensor(labels).float()

        pred_label = model(train_data)
        # print(pred_label[0])

        optimizer.zero_grad()
        l = loss(pred_label, labels)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    print("epoch:",i,"loss:", total_loss)

    if i==n_epoch-1:
        with th.no_grad():
            # print(batch['images'].shape)
            pred_rew = model(train_data)
        # print(pred_rew.shape, true_rew.shape)
        for b in range(batch_size):#pred_rew.shape[0]//20):
            # print(labels[b].shape)
            print(labels[b])
            print(pred_rew[b])
            print()