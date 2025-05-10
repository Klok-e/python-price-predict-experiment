import numpy as np
import pandas as pd
import torch
from backtesting import Backtest, Strategy
from line_profiler import profile
from sklearn.preprocessing import RobustScaler

from utils.util import preprocess_add_features, preprocess_make_ohlc_relative, scale_dataframe, \
    stop_loss_price, take_profit_price


class BuyAndHold(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def init(self):
        self.buy()

    @profile
    def next(self):
        pass


def create_buy_and_hold_strategy(data: pd.DataFrame, start: str, end: str):
    backtest_dataset = preprocess_add_features(data.loc[start:end])
    return Backtest(
        backtest_dataset,
        BuyAndHold,
        commission=0.001,
        exclusive_orders=True,
        cash=1_000_000,
    )


def create_backtest_model_with_data(
        model,
        data: pd.DataFrame,
        scaler,
        start: str,
        end: str,
        model_in_observations: int,
        print_actions=False,
        confidence_threshold=0.8
):
    skip_steps = 1024 + model_in_observations

    class NeuralNetStrat(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.current_order = None
            self.buy_price = None

        def init(self):
            pass

        # @profile
        def next(self):
            if self.equity <= 1:
                return

            if len(self.data) > skip_steps:
                df = self.data.df.iloc[-skip_steps:].copy()
                df.drop(columns=["Volume"], inplace=True)

                # cheating to improve performance
                preprocessed, _ = scale_dataframe(preprocess_make_ohlc_relative(df), scaler)
                observation = preprocessed.tail(model_in_observations).to_numpy(dtype=np.float32).reshape(1,
                                                                                                          model_in_observations,
                                                                                                          -1)
                curr_close = df.iloc[-1]["Close"]

                signal = model(torch.from_numpy(observation))
                if signal > confidence_threshold and self.buy_price is None:
                    self.current_order = self.buy()

                    self.buy_price = curr_close
                    if print_actions:
                        print(f"[{df.index.values[-1]}] bought at {self.buy_price}")

                if self.buy_price is not None and (stop_loss_price(self.buy_price, 0.4) >= curr_close or curr_close >= take_profit_price(self.buy_price, 0.4)):
                    self.sell()

                    if print_actions:
                        commission = 0.001
                        sell_fee = curr_close * (1 - commission)
                        buy_fee = self.buy_price * (1 + commission)
                        gain_from_trade_fee = (sell_fee - buy_fee) / buy_fee
                        print(
                            f"[{df.index.values[-1]}]"
                            + f" sold at {curr_close};"
                            + f" gain {gain_from_trade_fee};"
                            + f" equity {self.equity}"
                        )

                    self.buy_price = None

    backtest_dataset = preprocess_add_features(data.loc[start:end])
    # backtest_prepro_dataset, scaler = preprocess_scale(data.loc[start:end], scaler)
    return Backtest(
        backtest_dataset,
        NeuralNetStrat,
        commission=0.001,
        exclusive_orders=True,
        cash=1_000_000,
    )
