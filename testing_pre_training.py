from torch.optim import optimizer
from torch.utils.data import DataLoader
import cv2
from torch.nn.modules.rnn import LSTM
from general_utils import make_image_seq_strip, AttrDict
from sprites_datagen.rewards import *
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
import numpy as np
import torch as th
import torch.optim as optim
from tqdm import tqdm

from models import *
from util.radam import RAdam

def process_labels(rewards):
    true_rew = th.tensor([])
    for k in rewards.keys():
        for i in range(len(rewards[k])):
            true_rew = th.cat((true_rew, rewards[k][i]), 0)
    true_rew = true_rew.reshape(-1).unsqueeze(dim=1)
    return true_rew


reward_list = [VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
reward_list = [AgentXReward]
spec = AttrDict(
	resolution=64,
	max_seq_len=30,#30
	max_speed=0.05,      # total image range [0, 1]
	obj_size=0.2,       # size of objects, full images is 1.0
	shapes_per_traj=2,      # number of shapes per trajectory
	rewards= reward_list
)

dataset = MovingSpriteDataset(spec)
train_dataloader = DataLoader(dataset, batch_size=32)


n_epochs = 50
model = test_RewardGen()

optimizer = optim.Adam(model.parameters(), lr = 0.0005)
loss = th.nn.MSELoss()

for epoch in range(n_epochs):

    total_loss = 0

    for batch in tqdm(train_dataloader):

        train_data = batch['images'].reshape(-1,1,64,64)
        true_rew = process_labels(batch['rewards'])
        # print(train_data.shape, true_rew.shape)
        
        pred_rew = model(train_data)
        # print("Pred Rew Shape:", pred_rew.shape)
        # exit(0)
        # print()

        
        optimizer.zero_grad()
        # print(pred_rew.shape, true_rew.shape)
        l = loss(pred_rew, true_rew)
        l.backward()
        optimizer.step()

        total_loss += l.item()

    print("epoch:", epoch, ", loss:", total_loss)


for batch in train_dataloader:
    with th.no_grad():
        # print(batch['images'].shape)
        pred_rew = model(batch['images'].reshape(-1,1,64,64))
    true_rew = process_labels(batch['rewards'])
    # print(pred_rew.shape, true_rew.shape)
    for b in range(1):#pred_rew.shape[0]//20):
        # print(b)
        for i in range(20):
                print(true_rew[b*20+i], pred_rew[b*20+i])
        print()