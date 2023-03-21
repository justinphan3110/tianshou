
#source https://github.com/lerrytang/train-procgen-pfrl/blob/main/policies.py

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type, Union
import numpy as np

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, obs_space, num_outputs):

        super(ImpalaCNN, self).__init__()

        h, w, c = obs_space
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2],
                                   out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    def forward(self, obs):
        assert obs.ndim == 4
        x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x)
        return dist, value

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))



class DQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True), nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state
