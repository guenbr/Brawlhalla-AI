import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, num_actions=7, use_cnn=False, input_size=14,
                 input_channels=2, combined_data_size=8):
        super(ActorCritic, self).__init__()
        self.use_cnn = use_cnn

        if use_cnn:
            # CNN mode - visual processing
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            conv_out_size = self._get_conv_out((input_channels, 90, 160))
            feature_size = conv_out_size + combined_data_size
        else:
            # MLP mode - flat input only
            feature_size = input_size

        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, frames_or_data, combined_data=None):
        """
        CNN mode: forward(frames, combined_data)
        MLP mode: forward(flat_data)
        """
        if self.use_cnn:
            conv_out = self.conv(frames_or_data).view(frames_or_data.size()[0], -1)
            data_flat = combined_data.reshape(combined_data.size()[0], -1)
            x = torch.cat([conv_out, data_flat], dim=1)
        else:
            x = frames_or_data

        return self.actor(x), self.critic(x)