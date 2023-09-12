import random
import gymnasium as gym
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler, StandardScaler

OHLC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close"]
SEQUENCE_LENGTH = 128
PREDICTION_LENGTH = 32


class MultiScaler:
    def __init__(self, min_max: MinMaxScaler, std: StandardScaler):
        self.min_max = min_max
        self.std = std


def preprocess(df, scaler=None):
    # Apply percentage change only to OHLC columns
    df_pct = df[OHLC_COLUMNS].pct_change()

    # clamp
    df_pct[df_pct > 0.1] = 0.1
    df_pct[df_pct < -0.1] = -0.1

    # Concatenate the percentage-changed OHLC with the other columns
    df_all = pd.concat([df_pct, df.drop(columns=OHLC_COLUMNS)], axis=1)

    # Drop NA values (from pct_change operation)
    df_all = df_all.dropna()

    # Apply MinMax scaling to all columns
    if scaler is None:
        scaler = MultiScaler(MinMaxScaler(feature_range=(-1, 1)), StandardScaler())
        df_multi_scaled = scaler.min_max.fit_transform(scaler.std.fit_transform(df_all))
    else:
        df_multi_scaled = scaler.min_max.transform(scaler.std.transform(df_all))

    df_scaled = pd.DataFrame(df_multi_scaled,
                             columns=df_all.columns,
                             index=df_all.index)

    return df_scaled, scaler


def invert_preprocess(original_start, scaler: MultiScaler, df):
    original_start = original_start[OHLC_COLUMNS].to_numpy()
    # Invert MinMax scaling for all columns
    df_inv_scaled = pd.DataFrame(scaler.std.inverse_transform(scaler.min_max.inverse_transform(df)),
                                 columns=df.columns,
                                 index=df.index)

    # Recover the original OHLC values
    reversed_array = np.cumprod(1 + df_inv_scaled[OHLC_COLUMNS].to_numpy(), axis=0)
    reversed_array = reversed_array * original_start  # Scaling by the original_start to each element

    df_inv_scaled[OHLC_COLUMNS] = reversed_array

    return df_inv_scaled


def calculate_observation(df, model, scaler, buy_price):
    MODEL_INPUT_IN_OBSERVATION = 10

    df_with_features = add_features(df)

    original_start = df_with_features.iloc[-1]
    df_preprocessed, _ = preprocess(df_with_features, scaler)

    X = TimeSeries.from_dataframe(df_preprocessed[-SEQUENCE_LENGTH:])
    y = model.predict(PREDICTION_LENGTH, X, verbose=False, num_samples=8)

    y_mean = y.mean()

    y_inverted = invert_preprocess(original_start, scaler, y_mean.pd_dataframe())
    y_max_close = y_inverted.Close.mean()
    curr_close = original_start.Close
    prev_close = df_with_features.iloc[-2].Close

    y_std = y.std().pd_dataframe().Close.mean()
    predicted_gain = (y_max_close - curr_close) / curr_close
    current_gain = ((curr_close - buy_price) / buy_price) if buy_price is not None else 0
    last_32 = df_preprocessed.Close.iloc[-MODEL_INPUT_IN_OBSERVATION:].to_numpy().flatten()
    model_output = y_mean[:MODEL_INPUT_IN_OBSERVATION].pd_dataframe().Close.to_numpy().flatten()
    buy_status = 1 if buy_price is not None else 0
    observation = np.concatenate(
        [last_32, model_output, [predicted_gain], [buy_status], [y_std], [current_gain]]).astype(np.float32)

    return observation, curr_close, prev_close


class CustomEnv(gym.Env):
    SKIP_STEPS = 1200

    def __init__(self, dataset: pd.DataFrame, my_model: TFTModel, scaler: MultiScaler, episode_length=512,
                 commission=0.001, eval=False):
        super().__init__()

        self.eval = eval
        self.commission = commission

        # 0: hold; 1: buy; 2: sell
        self.action_space = spaces.Discrete(3)

        self.NUM_FEATURES = calculate_observation(dataset, my_model, scaler, False)[0].shape[0]
        print(f"obs length = {self.NUM_FEATURES}")

        # Update the observation space to include extra information
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.NUM_FEATURES,),
                                            dtype=np.float32)

        self.dataset = dataset
        self.my_model = my_model
        self.scaler = scaler
        self.episode_length = episode_length
        self.current_step = 0
        self.start_index = 0
        self.buy_price = None

    def step(self, action):
        self.current_step += 1

        # Calculate the current observation
        observation, curr_close, prev_close = self.calculate_observation(self.current_step)

        # Initialize reward and info
        reward = 0

        # Give small rewards or penalties for holding
        if self.buy_price is not None:
            change = (curr_close - prev_close) / prev_close
            if abs(change) > 0.001:
                if change > 0:
                    reward += 0.01  # small reward for holding when price increases
                elif change < 0:
                    reward -= 0.01  # small penalty for holding when price decreases

        # Action logic
        if self.buy_price is None:
            if action == 1:  # Buy
                self.buy_price = curr_close
                reward -= 0.02
        else:
            sell_fee = curr_close * (1 - self.commission)
            buy_fee = self.buy_price * (1 + self.commission)
            gain_from_trade_fee = (sell_fee - buy_fee) / buy_fee
            if action == 2:  # Sell
                self.buy_price = None
                # Large reward for a profitable sell
                reward += gain_from_trade_fee * 100

        info = {}
        terminated = self.current_step >= self.episode_length
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        if self.eval:
            random.seed(42)
        self.current_step = 0
        self.buy_price = None
        self.start_index = random.randint(self.SKIP_STEPS, len(self.dataset) - self.episode_length - 1)
        observation, _, _ = self.calculate_observation(self.current_step)
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass

    def calculate_observation(self, current_step):
        index_start = self.start_index + current_step - self.SKIP_STEPS
        index_end = self.start_index + current_step

        df = self.dataset.iloc[index_start:index_end].copy()
        observation, curr_close, prev_close = calculate_observation(df, self.my_model, self.scaler, self.buy_price)

        return observation, curr_close, prev_close


def add_features(df):
    df = df.copy()
    # add technical indicators to dataset
    df['SMA_256'] = df['Close'].rolling(window=256).mean()
    df['SMA_512'] = df['Close'].rolling(window=512).mean()
    df['SMA_1024'] = df['Close'].rolling(window=1024).mean()

    # convert SMA columns to distance in percentages from "Close"
    df['SMA_256'] = ((df['Close'] - df['SMA_256']) / df['SMA_256'])
    df['SMA_512'] = ((df['Close'] - df['SMA_512']) / df['SMA_512'])
    df['SMA_1024'] = ((df['Close'] - df['SMA_1024']) / df['SMA_1024'])

    # drop NaN rows resulting from the SMA calculations
    df = df.dropna()
    return df
