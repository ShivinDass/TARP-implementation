from gym import spaces
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common import policies
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
import copy

env = gym.make('Sprites-v0')
env = ExpandImgDimWrapper(env)
supervised_ppo = PolicyReplicate(env)
supervised_ppo.load_state_dict(th.load("saved_models/supervised_ppo_v0"))
# print(type(supervised_ppo))

policy_kwargs = dict(
    features_extractor_class = CustomFeatureExtractor, normalize_images=False#, net_arch = [{'pi':[32,32], 'vf':[32,32]}]
)
model = PPO(policies.ActorCriticPolicy, env, clip_range=0.2, learning_rate=1e-4, n_epochs=10, batch_size=120, n_steps=360, verbose=1, policy_kwargs = policy_kwargs)
model.policy.features_extractor.load_state_dict(supervised_ppo.features_extractor.state_dict())
model.policy.mlp_extractor.load_state_dict(supervised_ppo.mlp_extractor.state_dict())
model.policy.value_net.load_state_dict(supervised_ppo.value_net.state_dict())
model.policy.action_dist = copy.deepcopy(supervised_ppo.action_dist)
model.policy.action_net.load_state_dict(supervised_ppo.action_net.state_dict())
model.policy.log_std = supervised_ppo.log_std

# model.learn(total_timesteps=int(1e6))
# model.save("saved_models/best_model/ppo_encoder_v0")
# exit(0)
obs = env.reset()
plt.figure()
im = plt.imshow(env.render())

with th.no_grad():
    for i in range(1000):
        # action2, values2, log_prob2 = supervised_ppo.forward(th.tensor(obs).unsqueeze(dim=0), deterministic=True)
        # n_obs, reward, done, info = env.step(action[0])
        action, values, log_prob = model.policy.forward(th.tensor(obs).unsqueeze(dim=0), deterministic=False)
        a, _ = model.predict(obs, deterministic=True)
        n_obs, reward, done, info = env.step(a)

        print(action, values, log_prob)
        # print(action, reward)#, np.sqrt(np.sum(np.square(np.array(action)-np.array(action2[0])))))
        im.set_data(env.render())
        plt.show(block=False)

        obs = copy.deepcopy(n_obs)
        if done:
            print("done")
            obs = env.reset()
            prev_f = None
        l = input()