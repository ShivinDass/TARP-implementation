from gym import spaces
from numpy.core.fromnumeric import size
from stable_baselines3.ppo.ppo import PPO
import torch.nn as nn
import torch as th
import numpy as np
from models import *
import gym.spaces as spaces
import matplotlib.pyplot as plt
import sprites_env
import torch.optim as optim
from tqdm import tqdm
from env_wrappers import *

env = gym.make('Sprites-v0')
env = ExpandImgDimWrapper(env)
obs = env.reset()
plt.figure()
im = plt.imshow(env.render())

model = PolicyReplicate(env)
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
mse_loss = th.nn.MSELoss()

trained_model = PPO.load("saved_models/best_model/ppo_encoder_frozen_v0")

display = False
n_epochs = int(2e3)
batch_size = 840
mini_batch_size = 120
for epoch in range(n_epochs):
    obs = env.reset()
    
    if display:
        im.set_data(env.render())
        plt.show(block=False)

    batch = [obs]

    for steps in range(batch_size):
        action, _state = trained_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        batch.append(obs)

        if display:
            im.set_data(env.render())
            plt.show(block=False)
            l = input()

        if done:
            # print("done")
            obs = env.reset()
    
    np.random.shuffle(batch)
    total_loss = 0
    for i in range(0,batch_size, mini_batch_size):
        with th.no_grad():
            action_batch, values_batch, log_prob_batch = trained_model.policy.forward(th.tensor(batch[i:i+mini_batch_size]))
            # print("Action Shape:", action_batch.shape)
            # print("Values Shape:", values_batch.shape)
            # print("Log Prob Shape:", log_prob_batch.shape)
        
        action_pred, values_pred, log_prob_pred = model.forward(th.tensor(batch[i:i+mini_batch_size]))

        optimizer.zero_grad()
        loss = mse_loss(action_pred, action_batch) + mse_loss(values_pred, values_batch) + mse_loss(log_prob_pred, log_prob_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()/(batch_size/mini_batch_size)
    print("epoch:", epoch, ", loss:", total_loss)

th.save(model.state_dict(), "saved_models/supervised_ppo_v0")


with th.no_grad():
    for i in range(0, batch_size):
            action_batch, values_batch, log_prob_batch = trained_model.policy.forward(th.tensor(batch[i]).unsqueeze(dim=0))
            action_pred, values_pred, log_prob_pred = model.forward(th.tensor(batch[i]).unsqueeze(dim=0))

            print("Actions:", action_batch[0], action_pred[0])
            print("Values:", values_batch[0], values_pred[0])
            print("Log probs:", log_prob_batch[0], log_prob_pred[0])
            print()


    
    
