import time

from binance_historical_data import BinanceDataDumper

from util import __invert_preprocess, preprocess_scale, CustomEnv, OHLC_COLUMNS, preprocess_add_features, SEQUENCE_LENGTH, \
    PREDICTION_LENGTH, calculate_observation

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump=".",
    asset_class="spot",  # spot, um, cm
    data_type="klines",  # aggTrades, klines, trades
    data_frequency="1m",
)

data_dumper.dump_data(tickers=["NEARUSDT"])

import pandas as pd
import numpy as np
import os

filenames = next(os.walk("./spot/monthly/klines/NEARUSDT/1m"), (None, None, []))[2]  # [] if no file

columns = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore"
]

df = pd.DataFrame(columns=columns)

for f in filenames:
    new_df = pd.read_csv(f"./spot/monthly/klines/NEARUSDT/1m/{f}", header=None, names=columns)
    df = pd.concat([df, new_df])
df = df.sort_values(by="Open time")

dataset = df.loc[:, OHLC_COLUMNS].astype(np.float64)
dataset = pd.DataFrame(dataset.to_numpy(), columns=OHLC_COLUMNS)

dataset = preprocess_add_features(dataset)

preprocessed_dataset, scaler = preprocess_scale(dataset)

# transform dataset so that all transform-invert transform pairs are idempotent
dataset = __invert_preprocess(dataset.iloc[0], scaler, preprocessed_dataset)
dataset = preprocess_add_features(dataset)

from darts import TimeSeries
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

target_series = TimeSeries.from_dataframe(preprocessed_dataset)
target_train, target_val = target_series.split_after(0.95)

from darts.utils.likelihood_models import QuantileRegression
from darts.models import TFTModel

my_model = TFTModel(
    input_chunk_length=SEQUENCE_LENGTH,
    output_chunk_length=PREDICTION_LENGTH,
    hidden_size=128,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=512,
    n_epochs=1,
    add_relative_index=True,
    add_encoders=None,
    random_state=42,
    categorical_embedding_sizes={},
    likelihood=QuantileRegression(),
)

TFTMODEL_PATH = "model-weights-1694211368.0068371.pt"
my_model.load_weights(TFTMODEL_PATH)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = CustomEnv(dataset, my_model, scaler)
check_env(env)

# {'gamma': 0.8, 'ent_coef': 0.02, 'gae_lambda': 0.92}
rl_model = PPO("MlpPolicy", env,
               verbose=1,
               tensorboard_log="./tensorboard/",
               ent_coef=0.02,
               gae_lambda=0.92,
               gamma=0.8)
rl_model.set_parameters("rl_model-best_model2.zip")

with open('keys.txt', 'r') as f:
    api_key = f.readline().strip()
    secret_key = f.readline().strip()

from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

client = Client(api_key, secret_key)

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("NEARUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
klines = pd.DataFrame(klines, columns=columns)

# remove unused columns
klines = klines.loc[:, OHLC_COLUMNS].astype(np.float32)


class NeuralNetStrat:
    def __init__(self, rl_model, my_model, scaler):
        self.scaler = scaler
        self.my_model = my_model
        self.rl_model = rl_model
        self.buy_price = None
        self.paper_wallet = 1000.0  # Paper trading USD balance
        self.asset_balance = 0.0  # Paper trading NEAR balance

    def next(self, data):
        df = data.copy()
        observation, curr_close, _ = calculate_observation(df, self.my_model, self.scaler)

        action, _ = self.rl_model.predict(observation, deterministic=True)

        if self.buy_price is None:
            if action == 1:
                print("Buy signal")
                self.buy_price = curr_close
                self.asset_balance = self.paper_wallet / curr_close
                self.paper_wallet = 0.0

        else:
            if action == 2:
                print("Sell signal")
                commission = 0.001
                sell_fee = curr_close * (1 - commission)
                buy_fee = self.buy_price * (1 + commission)
                gain_from_trade_fee = (sell_fee - buy_fee) / buy_fee

                print(f"Gain: {gain_from_trade_fee}")
                self.paper_wallet = self.asset_balance * curr_close
                self.asset_balance = 0.0
                self.buy_price = None

        print(f"Paper Wallet Balance: {self.paper_wallet}")
        print(f"Asset Balance: {self.asset_balance}")


strat = NeuralNetStrat(rl_model, my_model, scaler)

while True:
    # Fetch new klines data
    klines = client.get_historical_klines("NEARUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
    klines = pd.DataFrame(klines, columns=columns)
    klines = klines.loc[:, OHLC_COLUMNS].astype(np.float32)

    # Call the strategy
    strat.next(klines)

    time.sleep(60)
