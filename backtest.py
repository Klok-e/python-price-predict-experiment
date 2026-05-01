import numpy as np
import pandas as pd
import torch
from backtesting import Backtest, Strategy
try:
    from line_profiler import profile
except ModuleNotFoundError:
    def profile(func):
        return func

from utils.experiment import DEFAULT_TRADING_CONTRACT, TradingContract
from utils.util import ensure_feature_columns, preprocess_make_ohlc_relative, scale_dataframe, \
    stop_loss_price, take_profit_price


def predict_model_probability(model, observation: np.ndarray):
    model_device = next(model.parameters()).device
    input_tensor = torch.from_numpy(observation).to(model_device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        return torch.sigmoid(logits).detach().cpu().reshape(-1)[0].item()


class BuyAndHold(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)

    def init(self):
        self.buy()

    @profile
    def next(self):
        pass


def create_buy_and_hold_strategy(
        data: pd.DataFrame,
        start: str,
        end: str,
        contract: TradingContract | None = None,
        cash=1_000_000,
):
    contract = contract or DEFAULT_TRADING_CONTRACT
    backtest_dataset = ensure_feature_columns(data.loc[start:end])
    return Backtest(
        backtest_dataset,
        BuyAndHold,
        commission=contract.commission,
        exclusive_orders=True,
        cash=cash,
    )


def create_backtest_model_with_data(
        model,
        data: pd.DataFrame,
        scaler,
        start: str,
        end: str,
        model_in_observations: int,
        print_actions=False,
        confidence_threshold=None,
        contract: TradingContract | None = None,
        cash=1_000_000,
):
    contract = contract or DEFAULT_TRADING_CONTRACT
    if confidence_threshold is None:
        confidence_threshold = contract.entry_probability_threshold()
    skip_steps = 1024 + model_in_observations

    class NeuralNetStrat(Strategy):
        def __init__(self, broker, data, params):
            super().__init__(broker, data, params)
            self.current_order = None
            self.buy_price = None
            self.pending_entry = False

        def init(self):
            pass

        # @profile
        def next(self):
            if self.equity <= 1:
                return

            if len(self.data) > skip_steps:
                if self.position and self.buy_price is None and len(self.trades) > 0:
                    self.buy_price = self.trades[-1].entry_price
                    self.pending_entry = False
                    if print_actions:
                        print(f"[{self.data.index[-1]}] entry filled at {self.buy_price}")

                df = self.data.df.iloc[-skip_steps:].copy()
                df.drop(columns=["Volume"], inplace=True, errors="ignore")

                # cheating to improve performance
                preprocessed, _ = scale_dataframe(preprocess_make_ohlc_relative(df), scaler)
                observation = preprocessed.tail(model_in_observations).to_numpy(dtype=np.float32).reshape(1,
                                                                                                          model_in_observations,
                                                                                                          -1)
                curr_close = df.iloc[-1]["Close"]

                signal = predict_model_probability(model, observation)
                if signal > confidence_threshold and self.buy_price is None and not self.pending_entry:
                    self.current_order = self.buy()
                    self.pending_entry = True
                    if print_actions:
                        print(f"[{df.index.values[-1]}] buy signal {signal:.4f}")

                if self.buy_price is not None and (
                        stop_loss_price(self.buy_price, contract.stop_loss_percent) >= curr_close
                        or curr_close >= take_profit_price(self.buy_price, contract.take_profit_percent)
                ):
                    self.position.close()

                    if print_actions:
                        commission = contract.commission
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
                    self.pending_entry = False

    backtest_dataset = ensure_feature_columns(data.loc[start:end])
    # backtest_prepro_dataset, scaler = preprocess_scale(data.loc[start:end], scaler)
    return Backtest(
        backtest_dataset,
        NeuralNetStrat,
        commission=contract.commission,
        exclusive_orders=True,
        cash=cash,
    )
