from pyexpat import features

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class PricePredictorModel(nn.Module):
    def __init__(self, window_size, feature_size, linear_arch=None):
        super(PricePredictorModel, self).__init__()

        if linear_arch is None:
            linear_arch = [64, 64]

        input_size = window_size * feature_size

        # Linear layers for the feedforward network
        self.linear_layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_f, out_f), nn.SiLU())
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


class PriceDataset(Dataset):
    def __init__(self, df_tickers, window_size):
        self.window_size = window_size
        self.scaled_data = [item[0] for item in df_tickers]
        self.labels = [item[2] for item in df_tickers]

    def __len__(self):
        return sum(len(data) - self.window_size + 1 for data in self.scaled_data)

    def __getitem__(self, idx):
        cumulative_length = 0
        for data, label in zip(self.scaled_data, self.labels):
            num_samples = len(data) - self.window_size + 1
            if idx < cumulative_length + num_samples:
                local_idx = idx - cumulative_length
                data_window = data.iloc[local_idx:local_idx + self.window_size].to_numpy(dtype=np.float32)
                label_value = label.iloc[local_idx + self.window_size - 1].to_numpy(np.float32)
                return data_window, label_value
            cumulative_length += num_samples


def train_supervised_model(df_tickers, window_size, tensorboard_log_dir, epochs=10, batch_size=4096, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_dataset = df_tickers[0][0]
    feature_size = first_dataset.shape[1]

    dataset = PriceDataset(df_tickers, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    writer = SummaryWriter(tensorboard_log_dir)

    model = PricePredictorModel(feature_size=feature_size, window_size=window_size, linear_arch=[512, 512, 512]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()

    writer.close()

    return model
