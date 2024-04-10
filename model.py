import math
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th

from util import OBS_PRICES_SEQUENCE, OBS_OTHER


class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, lstm_hidden_size=32, lstm_layers=1):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == OBS_PRICES_SEQUENCE:
                # Assuming sequence length is `seq_len` and feature number is `n_features`
                seq_len, n_features = subspace.shape
                extractors[key] = nn.LSTM(input_size=n_features, hidden_size=lstm_hidden_size, batch_first=True,
                                          num_layers=lstm_layers)
                total_concat_size += lstm_hidden_size  # hidden_dim is the LSTM output size
            elif key == OBS_OTHER:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs_data = observations[key]
            if key == OBS_PRICES_SEQUENCE:
                # LSTM expects input of shape (batch, seq_len, input_size)
                obs_data = obs_data.view(obs_data.size(0), -1, extractor.input_size)  # reshape if necessary
                out, _ = extractor(obs_data)
                # Using only the last output tensor (hidden state)
                encoded_tensor_list.append(out[:, -1, :])
            else:
                encoded_tensor_list.append(extractor(obs_data))
        # print(list(map(lambda x: x.shape, encoded_tensor_list)))
        return th.cat(encoded_tensor_list, dim=1)


class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, hidden_sizes=None):
        super().__init__(observation_space, features_dim=1)

        if hidden_sizes is None:
            hidden_sizes = [32]

        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            in_channels = get_flattened_obs_dim(subspace)
            layers = [nn.Flatten()]
            for out_channels in hidden_sizes:
                layers.append(nn.Linear(in_channels, out_channels))
                layers.append(nn.LeakyReLU())
                in_channels = out_channels

            extractors[key] = nn.Sequential(*layers)

            total_concat_size += hidden_sizes

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs_data = observations[key]
            encoded_tensor_list.append(extractor(obs_data))

        return th.cat(encoded_tensor_list, dim=1)


class SequenceCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,
                 cnn_channels=None,
                 kernel_size=None,
                 stride=None,
                 padding=None):
        super().__init__(observation_space, features_dim=1)

        if cnn_channels is None:
            cnn_channels = [16, 32]
        if kernel_size is None:
            kernel_size = 3
        if stride is None:
            stride = 1
        if padding is None:
            padding = 0

        self.extractors = nn.ModuleDict()
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == OBS_PRICES_SEQUENCE:
                seq_len, n_features = subspace.shape
                layers = []
                in_channels = n_features
                for out_channels in cnn_channels:
                    layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding))
                    layers.append(nn.LeakyReLU())
                    in_channels = out_channels
                layers.append(nn.Flatten())
                self.extractors[key] = nn.Sequential(*layers)
                # Calculate output size
                output_size = self._calculate_cnn_output_size(seq_len, cnn_channels, kernel_size, stride, padding)
                total_concat_size += output_size
            elif key == OBS_OTHER:
                self.extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        print(f"feature extractor total_concat_size = {total_concat_size}")

        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs_data = observations[key]
            if key == OBS_PRICES_SEQUENCE:
                # CNN expects input of shape (batch, channels, length), so transpose the sequence tensor
                obs_data = obs_data.transpose(1, 2)  # Swap seq_len and n_features
                encoded_tensor_list.append(extractor(obs_data))
            else:
                encoded_tensor_list.append(extractor(obs_data))

        return th.cat(encoded_tensor_list, dim=1)

    @staticmethod
    def _calculate_cnn_output_size(seq_len, cnn_channels, kernel_size, stride, padding):
        output_size = seq_len
        for _ in cnn_channels:
            output_size = (output_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return output_size * cnn_channels[-1]


class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, transformer_hidden_size=64, transformer_heads=4,
                 transformer_layers=1):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == OBS_PRICES_SEQUENCE:
                seq_len, n_features = subspace.shape
                transformer_layer = nn.TransformerEncoderLayer(
                    d_model=n_features,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_hidden_size,
                    batch_first=True
                )
                extractors[key] = nn.TransformerEncoder(
                    transformer_layer,
                    num_layers=transformer_layers
                )
                total_concat_size += n_features
                self.positional_encoding = nn.Parameter(self.create_positional_encoding(seq_len, n_features),
                                                        requires_grad=False)
            elif key == OBS_OTHER:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    @staticmethod
    def create_positional_encoding(seq_len, d_model):
        # Create positional encodings
        position = th.arange(seq_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = th.zeros(seq_len, d_model)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, observations: dict[th.Tensor]) -> th.Tensor:
        # print(list(map(lambda x: x[1].shape, observations.items())))
        # [torch.Size([10, 2]), torch.Size([10, 256, 12])] where first dim is batch size, 12 is amount of features
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs_data = observations[key]
            if key == OBS_PRICES_SEQUENCE:
                # Add positional encoding
                seq_len = obs_data.size(1)
                # Ensure positional encoding is added correctly
                obs_data += self.positional_encoding[:, :seq_len, :]
                out = extractor(obs_data)
                # print(out.shape)
                # torch.Size([10, 256, 12])
                # Make sure to take the last sequence output for each element in the batch
                encoded_tensor_list.append(out[:, -1])
            else:
                encoded_tensor_list.append(extractor(obs_data))
        # print(list(map(lambda x: x.shape,encoded_tensor_list)))
        # [torch.Size([10, 2]), torch.Size([10, 12])]
        return th.cat(encoded_tensor_list, dim=1)
