from datetime import datetime

import pandas as pd
from backtesting import Backtest, Strategy
from line_profiler import profile
from sklearn.preprocessing import MinMaxScaler

from util import SKIP_STEPS, calculate_observation, preprocess_add_features, preprocess_scale


def create_backtest_model_with_data(rl_model, data: pd.DataFrame, scaler: MinMaxScaler, start: str, end: str):
    class NeuralNetStrat(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.buy_price = None
            self.expected_gain = None

        def init(self):
            pass

        @profile
        def next(self):
            if self.equity <= 1:
                return

            if len(self.data) > SKIP_STEPS:
                df = self.data.df.iloc[-SKIP_STEPS:].copy()
                df.drop(columns=["Volume"], inplace=True)

                preprocessed, _ = preprocess_scale(df, scaler)
                observation, curr_close, _ = calculate_observation(preprocessed, df, self.buy_price)

                action, _ = rl_model.predict(observation, deterministic=True)
                if self.buy_price is None:
                    if action == 1:
                        self.buy()
                        self.buy_price = curr_close
                else:
                    if action == 2:
                        # commission = 0.001
                        # sell_fee = curr_close * (1 - commission)
                        # buy_fee = self.buy_price * (1 + commission)
                        # gain_from_trade_fee = (sell_fee - buy_fee) / buy_fee
                        # print(f"equity {self.equity}")
                        self.sell()
                        self.buy_price = None

    backtest_dataset = preprocess_add_features(data.loc[start:end])
    return Backtest(backtest_dataset, NeuralNetStrat, commission=.001, exclusive_orders=True, cash=1_000_000)
