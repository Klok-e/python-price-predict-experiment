from torch import nn


class PricePredictorModel(nn.Module):
    def __init__(self, window_size, feature_size, linear_arch=None):
        super(PricePredictorModel, self).__init__()

        if linear_arch is None:
            linear_arch = [64, 64]

        input_size = window_size * feature_size

        # Linear layers for the feedforward network
        self.linear_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_f, out_f), nn.LeakyReLU())
              for in_f, out_f in zip([input_size] + linear_arch[:-1], linear_arch)]
        )

        # Output layer to produce a single output
        self.output_layer = nn.Linear(linear_arch[-1], 1)
        # Sigmoid activation for the final output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        # Pass through the linear layers
        x = self.linear_layers(x)
        # Pass through the output layer
        x = self.output_layer(x)
        # Apply sigmoid activation
        return self.sigmoid(x)


class LSTMPricePredictorModel(nn.Module):
    def __init__(self, window_size, feature_size, hidden_size=128, num_layers=2, pre_lstm_arch=None,
                 post_lstm_arch=None):
        super(LSTMPricePredictorModel, self).__init__()

        if pre_lstm_arch is None:
            pre_lstm_arch = [64, 64]
        if post_lstm_arch is None:
            post_lstm_arch = [64, 64]

        # Linear layers before the LSTM
        self.pre_lstm_linear_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_f, out_f), nn.LeakyReLU())
              for in_f, out_f in zip([feature_size] + pre_lstm_arch[:-1], pre_lstm_arch)]
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=pre_lstm_arch[-1], hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)

        # Linear layers after the LSTM
        self.post_lstm_linear_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_f, out_f), nn.LeakyReLU())
              for in_f, out_f in zip([hidden_size] + post_lstm_arch[:-1], post_lstm_arch)]
        )

        # Output layer to produce a single output
        self.output_layer = nn.Linear(post_lstm_arch[-1], 1)
        # Sigmoid activation for the final output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass through the initial linear layers
        x = self.pre_lstm_linear_layers(x)
        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(x)
        # Use the last output of the LSTM
        x = lstm_out[:, -1, :]
        # Pass through the subsequent linear layers
        x = self.post_lstm_linear_layers(x)
        # Pass through the output layer
        x = self.output_layer(x)
        # Apply sigmoid activation
        return self.sigmoid(x)
