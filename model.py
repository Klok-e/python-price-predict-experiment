import math
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th


class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, lstm_hidden_size=32, lstm_layers=1):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "prices_sequence":
                # Assuming sequence length is `seq_len` and feature number is `n_features`
                seq_len, n_features = subspace.shape
                extractors[key] = nn.LSTM(input_size=n_features, hidden_size=lstm_hidden_size, batch_first=True,
                                          num_layers=lstm_layers)
                total_concat_size += lstm_hidden_size  # hidden_dim is the LSTM output size
            elif key == "other":
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: th.Tensor) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            obs_data = observations[key]
            if key == "prices_sequence":
                # LSTM expects input of shape (batch, seq_len, input_size)
                obs_data = obs_data.view(obs_data.size(0), -1, extractor.input_size)  # reshape if necessary
                out, _ = extractor(obs_data)
                # Using only the last output tensor (hidden state)
                encoded_tensor_list.append(out[:, -1, :])
            else:
                encoded_tensor_list.append(extractor(obs_data))
        # print(list(map(lambda x: x.shape, encoded_tensor_list)))
        return th.cat(encoded_tensor_list, dim=1)


class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, transformer_hidden_size=64, transformer_heads=4,
                 transformer_layers=1):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "prices_sequence":
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
            elif key == "other":
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = total_concat_size

    def create_positional_encoding(self, seq_len, d_model):
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
            if key == "prices_sequence":
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
