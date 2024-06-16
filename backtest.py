import pandas as pd
from backtesting import Backtest, Strategy
from line_profiler import profile
from sklearn.preprocessing import MinMaxScaler

from util import calculate_observation, preprocess_add_features, preprocess_scale


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
    rl_model,
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    start: str,
    end: str,
    model_in_observations: int,
):
    skip_steps = 1024 + model_in_observations

    class NeuralNetStrat(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
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
                preprocessed, _ = preprocess_scale(df, scaler)
                # preprocessed = backtest_prepro_dataset[self.data.index[0]:self.data.index[-1]]
                observation, curr_close, _ = calculate_observation(
                    preprocessed.tail(model_in_observations),
                    df.tail(model_in_observations),
                    self.buy_price,
                )

                action, _ = rl_model.predict(observation, deterministic=True)
                if action == 1 and self.buy_price is None:
                    self.buy()
                    self.buy_price = curr_close
                elif action == 2 and self.buy_price is not None:
                    self.position.close()
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
