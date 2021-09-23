from torch.utils.data import DataLoader
from sprites_datagen.rewards import *
from general_utils import make_image_seq_strip, AttrDict
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
from models import Encoder, EncoderDecoder
import torch as th

reward_list = [VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
reward_list = [VertPosReward]
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

enc = EncoderDecoder()

for batch in train_dataloader:
    train_data = batch['images'].reshape(-1,1,64,64)

    x = enc(train_data)
    print(x.shape)
    break