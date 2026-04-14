import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network used by the PPO agent to select actions and estimate state values

    Supports two input modes:
        - CNN mode: processes raw screen frames combined with flat game state data
        - Non-CNN mode: processes flat game state data only
    """

    def __init__(self, num_actions: int = 7, use_cnn: bool = False, input_size: int = 14,
                 input_channels: int = 2, combined_data_size: int = 8) -> None:
        """
        Initialize actor and critic networks based on input mode

        Args:
            num_actions (int): number of discrete actions the agent can take
            use_cnn (bool): whether to use CNN for visual frame processing
            input_size (int): size of flat input vector when in non-CNN mode
            input_channels (int): number of input channels for CNN frames
            combined_data_size (int): size of the flat game state vector appended to CNN output
        """
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
            # Non-CNN mode - API data only
            feature_size = input_size

        # Actor outputs a probability distribution over actions
        self.actor = nn.Sequential(
            nn.Linear(feature_size, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, num_actions),
            nn.Softmax(dim=-1)
        )

        # Critic estimates the value of the current state
        self.critic = nn.Sequential(
            nn.Linear(feature_size, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 512 if use_cnn else 64),
            nn.ReLU(),
            nn.Linear(512 if use_cnn else 64, 1)
        )

    def _get_conv_out(self, shape: tuple) -> int:
        """
        Calculate the flattened output size of the conv layers for a given input shape

        Args:
            shape (tuple): input shape as (channels, height, width)

        Returns:
            int: number of features output by the conv layers after flattening
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, frames_or_data: torch.Tensor, combined_data: torch.Tensor | None = None) -> \
            tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor and critic networks

        Args:
            frames_or_data (torch.Tensor): screen frames in CNN mode, or flat game state in non-CNN mode
            combined_data (torch.Tensor | None): flat game state vector appended to CNN output, None in non-CNN mode

        Returns:
            tuple containing:
                - action_probs (torch.Tensor): probability distribution over actions
                - value (torch.Tensor): estimated state value from the critic
        """
        if self.use_cnn:
            # Flatten conv output and concatenate with game state vector
            conv_out = self.conv(frames_or_data).view(frames_or_data.size()[0], -1)
            data_flat = combined_data.reshape(combined_data.size()[0], -1)
            x = torch.cat([conv_out, data_flat], dim=1)
        else:
            x = frames_or_data

        return self.actor(x), self.critic(x)
