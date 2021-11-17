from numpy.lib.type_check import imag
from sklearn.linear_model import LinearRegression
import torch as th
from sprites_datagen.rewards import *
from models import *
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
from general_utils import make_image_seq_strip, AttrDict
from torch.utils.data import DataLoader
import numpy as np

# reward_list = [VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
reward_list = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
spec = AttrDict(
	resolution=64,
	max_seq_len=30,#30
	max_speed=0.05,      # total image range [0, 1]
	obj_size=0.2,       # size of objects, full images is 1.0
	shapes_per_traj=4,      # number of shapes per trajectory
	rewards= reward_list
)

dataset = MovingSpriteDataset(spec)
train_dataloader = DataLoader(dataset, batch_size=32)

reward_encoder = Encoder()
reward_encoder.load_state_dict(th.load("saved_models/encoder_model"))

image_encoder = Encoder()
image_encoder.load_state_dict(th.load("saved_models/autoencoder_encoder"))

for b in train_dataloader:
    batch = b
with th.no_grad():
    print(batch['images'].shape)
    encoded_r = np.array(reward_encoder.forward(batch['images'].reshape(-1,1,64,64)))
    encoded_i = np.array(image_encoder.forward(batch['images'].reshape(-1,1,64,64)))

print(encoded_r.shape)
print(encoded_i.shape)
# exit(0)
for k in batch['rewards'].keys():
    rew = np.array(batch['rewards'][k].reshape(-1))
    
    reg_r = LinearRegression().fit(encoded_r, rew)
    reg_i = LinearRegression().fit(encoded_i, rew)
    
    v = (np.square(rew-rew.mean())).sum()

    print(k)
    print("Reward:", v*(1-reg_r.score(encoded_r, rew)))
    print("Image:", v*(1-reg_i.score(encoded_i, rew)))
    print()