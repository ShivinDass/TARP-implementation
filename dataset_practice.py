import cv2
from torch.nn.modules.rnn import LSTM
from general_utils import make_image_seq_strip, AttrDict
from sprites_datagen.rewards import *
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, MovingSpriteDataset
import numpy as np
import torch as th

from models import *

spec = AttrDict(
	resolution=64,
	max_seq_len=30,#30
	max_speed=0.05,      # total image range [0, 1]
	obj_size=0.2,       # size of objects, full images is 1.0
	shapes_per_traj=2,      # number of shapes per trajectory
	rewards=[VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward],
)

# gen = DistractorTemplateMovingSpritesGenerator(spec)
# traj = gen.gen_trajectory()
# img = make_image_seq_strip([traj.images[None, :, None].repeat(3, axis=2).astype(np.float32)], sep_val=255.0).astype(np.uint8)
# cv2.imwrite("test.png", img[0].transpose(1, 2, 0))
# exit(0)

dataset = MovingSpriteDataset(spec)
for _ in range(1):
	print("Images Shape:",dataset[0].images.shape,
		"\nStates Shape:", dataset[0].states.shape,
		"\n\nRewards Shape:-")

	batch_size = 32
	true_rew = []
	for k in dataset[0].rewards.keys():
		print(k, dataset[0].rewards[k].shape)
		for i in range(batch_size):
			true_rew.append(dataset[i].rewards[k])
	print()
	true_rew = th.tensor(true_rew).reshape(-1).unsqueeze(dim=1)
	print("True Out Shape:", true_rew.shape)
	
	data = []
	for i in range(batch_size):
		data.append(dataset[i].images)# = np.concatenate(data, dataset[i].images)
	data = th.tensor(data)
	print("Data Shape:",data.shape)
	
	print()
	model = RewardGen(len(dataset[0].rewards.keys()))
	pred_rew = model(data)

	for sd in model.lstm_encoder.encoder.state_dict():
		print(sd,":", model.lstm_encoder.encoder.state_dict()[sd].size())
	loss = th.nn.MSELoss()
	l = loss(pred_rew, true_rew)
	l.backward()

	print(loss)
	print(l)

	# print()
	# enc = Encoder()
	# print(enc(th.tensor(data.images)).shape)


