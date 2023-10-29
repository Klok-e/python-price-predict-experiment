from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import torch as th


class CustomExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

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

        return th.cat(encoded_tensor_list, dim=1)
