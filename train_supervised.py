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
    """
    Produces overlapping (or strided) sliding windows across
    every ticker’s dataframe.

    stride = 1  → classic “next-timestep” overlap
    stride = k  → windows start every k timesteps
    """
    def __init__(self, df_tickers, window_size: int, stride: int = 1):
        self.window_size = window_size
        self.stride = stride
        self.scaled_data = [item[0] for item in df_tickers]
        self.labels = [item[2] for item in df_tickers]

    def __len__(self):
        return sum(
            (len(data) - self.window_size) // self.stride + 1
            for data in self.scaled_data
        )

    def __getitem__(self, idx):
        cumulative_length = 0
        for data, label in zip(self.scaled_data, self.labels):
            num_samples = (len(data) - self.window_size) // self.stride + 1
            if idx < cumulative_length + num_samples:
                local_idx = (idx - cumulative_length) * self.stride
                data_window = data.iloc[local_idx:local_idx + self.window_size].to_numpy(dtype=np.float32)
                label_value = label.iloc[local_idx + self.window_size - 1].to_numpy(dtype=np.float32)
                return data_window, label_value
            cumulative_length += num_samples

        return None


def calculate_class_weights(labels):
    labels = labels.astype(np.int32, copy=False)
    class_counts = np.bincount(labels)
    class_weights = len(labels) / class_counts
    return class_weights


def calculate_sample_weights(df_tickers, window_size: int, stride: int = 1):
    # pick labels at the same stride used by the dataset
    all_labels = np.concatenate([
        ticker[2].iloc[window_size - 1 :: stride] for ticker in df_tickers
    ]).reshape(-1)

    # Calculate class weights
    class_weights = calculate_class_weights(all_labels)
    return np.array([class_weights[int(label)] for label in all_labels])


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
                           continue_training=True, test_prediction_threshold=0.5,
                           window_stride: int = 8, random_sampler_samples_percent=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_size = df_tickers_train[0][0].shape[1]

    # Create datasets
    train_dataset = PriceDataset(df_tickers_train, window_size, stride=window_stride)
    test_dataset = PriceDataset(df_tickers_test, window_size, stride=window_stride)

    sample_weights_train = calculate_sample_weights(df_tickers_train, window_size, stride=window_stride)
    sampler_train = WeightedRandomSampler(sample_weights_train,
                                          int(random_sampler_samples_percent * len(train_dataset)))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    log_dir = f"{computed_data_dir}/tensorboard/"
    create_dir_if_not_exists(log_dir)

    with SummaryWriter(f"{log_dir}{model_name}{len(os.listdir(log_dir))}") as writer:
        model = model_type(feature_size=feature_size, window_size=window_size, **model_kwargs).to(device)

        # Load pre-trained model if continue_training is True
        model_save_path = f"{computed_data_dir}/supervised_model/{model_name}.pth"
        if continue_training:
            model = load_model(model, model_save_path)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            epoch_train_loss = 0
            interval_train_labels = []
            interval_train_predictions = []

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, labels)

                train_loss += loss.item()
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > test_prediction_threshold).float()
                    interval_train_labels.extend(labels.cpu().numpy())
                    interval_train_predictions.extend(preds.cpu().numpy())

                if (batch_idx + 1) % log_interval == 0:
                    # -------- train accuracy ----------
                    train_accuracy = (np.array(interval_train_predictions) == np.array(interval_train_labels)).mean()
                    writer.add_scalar(
                        "train/Accuracy",
                        train_accuracy,
                        epoch * len(train_dataloader) + batch_idx
                    )
                    # reset accumulators for next interval
                    interval_train_labels.clear()
                    interval_train_predictions.clear()

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
                    all_probs = []

                    with torch.no_grad():
                        for inputs, labels in test_dataloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logits = model(inputs)
                            loss = criterion(logits, labels)
                            total_test_loss += loss.item()

                            # Calculate predictions
                            probs = torch.sigmoid(logits)
                            predictions = (probs > test_prediction_threshold).float()

                            # Store labels, probabilities, and predictions
                            all_labels.extend(labels.cpu().numpy())
                            all_probs.extend(probs.cpu().numpy())
                            all_predictions.extend(predictions.cpu().numpy())

                    # Calculate additional metrics
                    test_accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
                    test_roc_auc = roc_auc_score(all_labels, all_probs)
                    test_precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
                    test_recall = recall_score(all_labels, all_predictions, average='binary')
                    test_f1 = f1_score(all_labels, all_predictions, average='binary')

                    # Log metrics
                    writer.add_scalar("test/Loss", total_test_loss / len(test_dataloader),
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/Accuracy", test_accuracy,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/ROC AUC", test_roc_auc,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/Precision", test_precision,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/Recall", test_recall,
                                      epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar("test/F1 Score", test_f1,
                                      epoch * len(train_dataloader) + batch_idx)

                    if save_model:
                        create_dir_if_not_exists(model_save_path)
                        torch.save(model.state_dict(), model_save_path)

                    model.train()

            scheduler.step(epoch_train_loss / len(train_dataloader))

            print(f"epoch {epoch} ended; epoch training loss: {epoch_train_loss / len(train_dataloader)}")

    return model