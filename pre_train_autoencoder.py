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
import matplotlib.pyplot as plt

def process_labels(rewards):
    true_rew = th.tensor([])
    for k in rewards.keys():
        for i in range(len(rewards[k])):
            true_rew = th.cat((true_rew, rewards[k][i]), 0)
    true_rew = true_rew.reshape(-1).unsqueeze(dim=1)
    return true_rew


reward_list = [VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
reward_list = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
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

model = EncoderDecoder()
model.load_state_dict(th.load("saved_models/full_autoencoder"))
with th.no_grad():
    fig, (ax1,ax2) = plt.subplots(1,2)
    for batch in train_dataloader:
        test_data = th.tensor(batch['images']).reshape(-1,1,64,64)
        for img in test_data:
            ax1.imshow(img[0])
            ax2.imshow(model(th.unsqueeze(img,0))[0][0])
            plt.show(block=False)
            l = input()
exit(0)

n_epochs = 300
model = EncoderDecoder()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

loss = th.nn.BCELoss()

# test_img = th.unsqueeze(th.tensor(dataset[0]['images'][0]).clone().detach(), 0)
for epoch in range(n_epochs):

    total_loss = 0

    for batch in tqdm(train_dataloader):

        train_data = th.tensor(batch['images']).reshape(-1,1,64,64)
        pred_data = model(train_data)
        
        optimizer.zero_grad()
        l = loss(pred_data, train_data)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    
    # if epoch%10==0:
    #     with th.no_grad():
    #         plt.imshow(model(test_img)[0][0])
    #         plt.show(block=False)

    print("epoch:", epoch, ", loss:", total_loss)

th.save(model.state_dict(), "saved_models/full_autoencoder")
th.save(optimizer.state_dict(), "saved_models/full_autoencoder_optim")
th.save(model.encoder.state_dict(), "saved_models/autoencoder_encoder")
