import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
                label_value = label.iloc[local_idx + self.window_size - 1].to_numpy(dtype=np.float32)
                return data_window, label_value
            cumulative_length += num_samples

        return None


# def calculate_class_weights(labels):
#     class_counts = np.bincount(labels)
#     class_weights = len(labels) / class_counts
#     return class_weights
#
#
# def calculate_sample_weights(df_tickers, window_size):
#     all_labels = np.concatenate([ticker[2][window_size - 1:] for ticker in df_tickers]).reshape(-1)
#
#     # Calculate class weights
#     class_weights = calculate_class_weights(all_labels)
#     return np.array([class_weights[int(label)] for label in all_labels])


def load_model(model, model_path):
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f"No pre-trained model found at {model_path}.")
    return model


def train_supervised_model(model_type, model_kwargs, df_tickers_train, df_tickers_test, window_size,
                           computed_data_dir, model_name, epochs=10,
                           batch_size=4096, learning_rate=0.0001, log_interval=100, save_model=False,
                           continue_training=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_size = df_tickers_train[0][0].shape[1]

    # Create datasets
    train_dataset = PriceDataset(df_tickers_train, window_size)
    test_dataset = PriceDataset(df_tickers_test, window_size)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    log_dir = f"{computed_data_dir}/tensorboard/"
    create_dir_if_not_exists(log_dir)

    with SummaryWriter(f"{log_dir}{model_name}{len(os.listdir(log_dir))}") as writer:
        model = model_type(feature_size=feature_size, window_size=window_size, **model_kwargs).to(device)

        # Load pre-trained model if continue_training is True
        model_save_path = f"{computed_data_dir}/supervised_model/{model_name}.pth"
        if continue_training:
            model = load_model(model, model_save_path)

        # Change to regression loss: Mean Squared Error
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            epoch_train_loss = 0

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                train_loss += loss.item()
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                if (batch_idx + 1) % log_interval == 0:
                    writer.add_scalar("train/Loss", train_loss / log_interval,
                                      epoch * len(train_dataloader) + batch_idx)

                    writer.add_scalar("train/Learning Rate", scheduler.get_last_lr()[0],
                                      epoch * len(train_dataloader) + batch_idx)

                    train_loss = 0

                    # Evaluate on test data
                    model.eval()
                    total_test_loss = 0
                    all_labels = []
                    all_predictions = []

                    with torch.no_grad():
                        for inputs, labels in test_dataloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            total_test_loss += loss.item()

                            # Store labels and predictions for metrics
                            all_labels.extend(labels.cpu().numpy())
                            all_predictions.extend(outputs.cpu().numpy())

                    # Calculate regression metrics
                    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                    mse = mean_squared_error(all_labels, all_predictions)
                    mae = mean_absolute_error(all_labels, all_predictions)
                    r2 = r2_score(all_labels, all_predictions)

                    # For monitoring trade signal distribution
                    prediction_histogram = np.histogram(all_predictions, bins=10, range=(0.0, 1.0))[0]

                    # Log metrics
                    writer.add_scalar("test/Loss", total_test_loss / len(test_dataloader),
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/MSE", mse,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/MAE", mae,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/R2", r2,
                                      epoch * len(train_dataloader) + batch_idx)

                    # Log prediction distribution
                    for i, count in enumerate(prediction_histogram):
                        writer.add_scalar(f"test/Pred_Bin_{i * 0.1:.1f}-{(i + 1) * 0.1:.1f}", count,
                                          epoch * len(train_dataloader) + batch_idx)

                    if save_model:
                        create_dir_if_not_exists(model_save_path)
                        torch.save(model.state_dict(), model_save_path)

                    model.train()

            scheduler.step(epoch_train_loss / len(train_dataloader))

            print(f"epoch {epoch} ended; epoch training loss: {epoch_train_loss / len(train_dataloader)}")

    return model
