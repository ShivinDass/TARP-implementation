from sprites_datagen.rewards import Reward
import torch as th
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

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


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.in_channels = 1
        self.input_dim = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0), nn.LeakyReLU()
        )

        self.fc1 = nn.Linear(in_features=128, out_features=64)

    def forward(self, input):
        x = self.encoder(input)
        # print(x.shape)
        x = th.flatten(x, start_dim=1)
        # print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        return x

class LSTM_Encoder(nn.Module):
    def __init__(self):
        super(LSTM_Encoder, self).__init__()

        self.encoder = Encoder()

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
        x = x.reshape(batch_size, seq_len, -1)
        # print(x.shape)

        out, hidden = self.lstm(x)
        # print("Out:", out.shape)
        return out

class RewardGen(nn.Module):
    def __init__(self, n_rewards):
        super(RewardGen, self).__init__()

        self.n_rewards = n_rewards
        self.lstm_encoder = LSTM_Encoder()

        self.reward_mlp = []
        for _ in range(n_rewards):
            self.reward_mlp.append(MLP(input_dim=64, output_dim=1))
    
    def forward(self, input):
        x = self.lstm_encoder(input)
        h_dim = x.shape[-1]

        x = x.reshape(-1, h_dim)
        
        out = self.reward_mlp[0](x)
        # print(out.shape)
        for i in range(1,self.n_rewards):
            out = th.cat((out,self.reward_mlp[i](x)),0)
        # print(out.shape)

        return out