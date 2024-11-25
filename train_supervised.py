import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils.util import create_dir_if_not_exists


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


def train_supervised_model(model_type, model_kwargs, df_tickers_train, df_tickers_test, window_size,
                           computed_data_dir, epochs=10,
                           batch_size=4096, learning_rate=0.01, log_interval=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_dataset = df_tickers_train[0][0]
    feature_size = first_dataset.shape[1]

    # Create datasets and dataloaders
    train_dataset = PriceDataset(df_tickers_train, window_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = PriceDataset(df_tickers_test, window_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    log_dir = f"{computed_data_dir}/tensorboard/"
    create_dir_if_not_exists(log_dir)
    writer = SummaryWriter(f"{log_dir}run{len(os.listdir(log_dir))}")

    model = model_type(feature_size=feature_size, window_size=window_size, **model_kwargs).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    for epoch in range(epochs):
        model.train()
        total_test_loss = 0
        train_loss = 0

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % log_interval == 0:
                writer.add_scalar("Loss/train", train_loss / log_interval,
                                  epoch * len(train_dataloader) + batch_idx)

                writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0],
                                  epoch * len(train_dataloader) + batch_idx)

                train_loss = 0

                # Evaluate on test data
                model.eval()
                total_test_loss = 0
                correct_predictions = 0
                total_samples = 0
                with torch.no_grad():
                    for inputs, labels in test_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        total_test_loss += loss.item()

                        # Calculate accuracy
                        predictions = (outputs > 0.5).float()
                        correct_predictions += (predictions == labels).sum().item()
                        total_samples += labels.size(0)

                test_accuracy = correct_predictions / total_samples
                writer.add_scalar("Loss/test", total_test_loss / len(test_dataloader),
                                  epoch * len(train_dataloader) + batch_idx)
                writer.add_scalar("Accuracy/test", test_accuracy,
                                  epoch * len(train_dataloader) + batch_idx)

                model_save_path = f"{computed_data_dir}/supervised_model/epoch_{epoch}_batch_{batch_idx}_test_accuracy_{test_accuracy}.pth"
                create_dir_if_not_exists(model_save_path)
                torch.save(model.state_dict(), model_save_path)

                model.train()

        scheduler.step(total_test_loss / len(test_dataloader))

        print(f"epoch {epoch} ended")

    writer.close()

    return model
