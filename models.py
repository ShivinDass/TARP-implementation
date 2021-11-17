from copy import deepcopy
from torch.nn.modules import activation
from sprites_datagen.rewards import Reward
import torch as th
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.preprocessing import preprocess_obs
import numpy as np
import gym

class MLP32(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP32, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.mlp(input)

class MLP64(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP64, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.output_dim),
            nn.ReLU()
        )

    def forward(self, input):
        return self.mlp(input)


class Encoder(nn.Module):

    def __init__(self, activation_fn = nn.LeakyReLU):
        super(Encoder, self).__init__()

        self.in_channels = 1
        self.input_dim = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1), activation_fn(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1), activation_fn(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1), activation_fn(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1), activation_fn(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), activation_fn(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0), activation_fn()
        )
        
        self.fc1 = nn.Linear(in_features=128, out_features=64)

    def forward(self, input):

        x = self.encoder(input)
        x = th.flatten(x, start_dim=1)
        x = th.sigmoid(self.fc1(x))
        return x

class Encoder2(nn.Module):

    def __init__(self, activation_fn = nn.LeakyReLU):
        super(Encoder2, self).__init__()

        self.in_channels = 1
        self.input_dim = 64

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0)
        
        self.activation = activation_fn()

        self.fc1 = nn.Linear(in_features=128, out_features=64)

    def forward(self, input):
        with th.no_grad():
            x = self.activation(self.conv1(input))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))
            x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x))
         
        x = th.flatten(x, start_dim=1)
        x = th.sigmoid(self.fc1(x))
        return x



class LSTM_Encoder(nn.Module):
    def __init__(self):
        super(LSTM_Encoder, self).__init__()

        self.encoder = Encoder()
        self.mlp = MLP64(input_dim=64, output_dim=64)

        self.input_size = 64
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = 1, batch_first = True)

    def forward(self, input):
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        
        x = input.reshape(batch_size*seq_len, 1, 64, 64)
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        x = self.mlp(x)
        # print(x.shape)
        x = x.reshape(batch_size, seq_len, -1)
        # print(x.shape)

        out, hidden = self.lstm(x)
        # print("Out:", out.shape)
        return out

class RewardGen(nn.Module):
    def __init__(self, n_rewards):
        super(RewardGen, self).__init__()
        assert n_rewards>0

        self.lstm_encoder = LSTM_Encoder()

        self.n_rewards = n_rewards
        self.reward_mlp = nn.ModuleList()
        for _ in range(n_rewards):
            self.reward_mlp.append(MLP32(input_dim=64, output_dim=1))


    def forward(self, input):
        x = self.lstm_encoder(input)
        h_dim = x.shape[-1]

        x = x.reshape(-1, h_dim)
        
        out = self.reward_mlp[0](x)
        for i in range(1,self.n_rewards):
            out = th.cat((out,self.reward_mlp[i](x)),0)
        # out = self.reward_mlp[0](x)
        return out


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = Encoder()
        self.flatten = nn.Flatten()
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        with th.no_grad():
            return self.flatten(self.encoder(observations))

class CustomFeatureExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.flatten = nn.Flatten()
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(self.encoder1(observations)), self.flatten(self.encoder2(observations))

class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        
        self.encoder = Encoder()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, input):
        # batch_size = input.shape[0]
        # seq_len = input.shape[1]
        # x = input.reshape(batch_size*seq_len, 1, 64, 64)
        
        x = self.encoder(input)
        # print(x.shape)

        x = x.reshape(-1, 64, 1, 1)
        x = self.decoder(x)
        # print(x.shape)
        return x



class test_RewardGen(nn.Module):

    def __init__(self, n_rewards):
        super(test_RewardGen, self).__init__()
        assert n_rewards>0
        self.encoder = Encoder()

        self.n_rewards = n_rewards
        self.reward_mlp = nn.ModuleList()
        for _ in range(n_rewards):
            self.reward_mlp.append(MLP32(input_dim=64, output_dim=1))
        # self.mlp = MLP32(input_dim=64, output_dim=1)

    
    def forward(self, input):
        x = self.encoder(input)
        
        out = self.reward_mlp[0](x)
        # print(out.shape)
        for i in range(1,self.n_rewards):
            out = th.cat((out,self.reward_mlp[i](x)),0)
        # out = self.mlp(x)        
        return out


class PolicyReplicate(nn.Module):

    def __init__(self, env, features_dim = 64, net_arch = [{'pi':[64,64], 'vf':[64,64]}], activation_fn = nn.Tanh):
        super(PolicyReplicate, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.features_extractor = CustomFeatureExtractor(observation_space = self.observation_space, features_dim = features_dim)

        self.mlp_extractor = MlpExtractor(feature_dim = features_dim,
                                        net_arch = net_arch,
                                        activation_fn = activation_fn)
        
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.action_dist = DiagGaussianDistribution(action_dim=int(np.prod(self.action_space.shape)))
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi, log_std_init=0.0)

    def forward(self, obs, deterministic: bool = False):
        # obs = th.tensor(obs)
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=False)
        features = self.features_extractor(preprocessed_obs)

        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_sde = latent_pi

        values = self.value_net(latent_vf)
        mean_actions = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob