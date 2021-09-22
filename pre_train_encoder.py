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

def process_labels(rewards):
    true_rew = th.tensor([])
    for k in rewards.keys():
        for i in range(len(rewards[k])):
            true_rew = th.cat((true_rew, rewards[k][i]), 0)
    true_rew = true_rew.reshape(-1).unsqueeze(dim=1)
    return true_rew


reward_list = [VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
reward_list = [VertPosReward]
spec = AttrDict(
	resolution=8,
	max_seq_len=30,#30
	max_speed=0.05,      # total image range [0, 1]
	obj_size=0.2,       # size of objects, full images is 1.0
	shapes_per_traj=2,      # number of shapes per trajectory
	rewards= reward_list
)

dataset = MovingSpriteDataset(spec)
train_dataloader = DataLoader(dataset, batch_size=32)
# th.save(train_dataloader, "data/pre_training_data32.pth")
print(dataset[0]['images'][0])
exit(0)


# train_dataloader = th.load("data/pre_training_data.pth")
n_epochs = 1
model = RewardGen(len(spec['rewards']))
# checkpoint = th.load("saved models/full_reward_gen_model")
# model.load_state_dict(checkpoint)

optimizer = optim.Adam(model.parameters(), lr = 0.0005)
loss = th.nn.MSELoss()
# checkpoint2 = th.load("saved models/full_reward_optim")
# optimizer.load_state_dict(checkpoint2)
# print(optimizer.state_dict)


# print(model.lstm_encoder.encoder.state_dict)

# model.train()
for epoch in range(n_epochs):

    total_loss = 0

    for batch in tqdm(train_dataloader):

        train_data = batch['images']
        # print("Train Data Shape:",train_data.shape)
        true_rew = process_labels(batch['rewards'])
        # print("True Out Shape:", true_rew.shape)
        
        pred_rew = model(train_data)
        # print("Pred Rew Shape:", pred_rew.shape)
        # print()

        
        optimizer.zero_grad()
        l = loss(pred_rew, true_rew)
        l.backward()
        optimizer.step()

        total_loss += l.item()

    print("epoch:", epoch, ", loss:", total_loss)

# for param in model.lstm_encoder.encoder.parameters():
#     print(param)
# print("\n\n")

th.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict()}, "saved models/full_reward_gen_model")
# th.save(model.state_dict(), "saved models/full_reward_gen_model")
# th.save(optimizer.state_dict(), "saved models/full_reward_optim")
th.save(model.lstm_encoder.encoder.state_dict(), "saved models/encoder_model")
