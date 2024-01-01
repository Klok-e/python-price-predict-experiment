import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data, create_buy_and_hold_strategy
from trading_metrics import calculate_metrics
from util import download_and_process_data_if_available

# rl_model = PPO.load("cherry-picked-best-models/rl-model-best1.pt")
# rl_model = PPO.load("cherry-picked-best-models/best_model1.zip")
# rl_model = PPO.load("cherry-picked-best-models/best_model.zip")
rl_model = PPO.load("cherry-picked-best-models/hs128_lstm2_net[256, 256, 256]_ws64_2499750_steps.zip")

df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")


def run_backtest_on_all_tickers(strat_name, create_strategy_func):
    sum_equity = None
    for _, df, scaler, name in df_tickers:
        model_in_observations = 128
        start = "2023-09-01"
        end = "2023-10-01"

        t = time.time()
        print(f"backtest for {name} started")

        skip_steps = 1024 + model_in_observations
        bt = create_strategy_func(df, scaler, model_in_observations, start, end)
        _ = bt.run()

        equity = bt._results._equity_curve["Equity"].iloc[skip_steps:]
        if sum_equity is None:
            sum_equity = equity
        else:
            sum_equity += equity

        print(f"backtest for {name} finished; time taken: {time.time() - t}")
    start_cash = 1_000_000
    tickers_count = len(df_tickers)

    y = (sum_equity - start_cash * tickers_count) / (start_cash * tickers_count)
    plt.plot(y, label=strat_name)

    return calculate_metrics(sum_equity)


model_metrics = run_backtest_on_all_tickers("Buy and Hold",
                                            lambda df, scaler, model_in_observations, start,
                                                   end: create_buy_and_hold_strategy(
                                                df,
                                                start,
                                                end))
baseline_metrics = run_backtest_on_all_tickers("Model",
                                               lambda df, scaler, model_in_observations, start,
                                                      end: create_backtest_model_with_data(
                                                   rl_model,
                                                   df,
                                                   scaler,
                                                   start,
                                                   end,
                                                   model_in_observations))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.tight_layout()

plt.show()

avg_metrics = model_metrics
print()
print("Model metrics:")
print(f"cumulative_return={avg_metrics[0]:.4f}, "
      f"max_earning_rate={avg_metrics[1]:.4f}, "
      f"maximum_pullback={avg_metrics[2]:.4f}, "
      f"average_profitability_per_trade={avg_metrics[3]:.4f}, "
      f"sharpe_ratio={avg_metrics[4]:.4f}")
print()

avg_metrics = baseline_metrics
print("Baseline metrics:")
print(f"cumulative_return={avg_metrics[0]:.4f}, "
      f"max_earning_rate={avg_metrics[1]:.4f}, "
      f"maximum_pullback={avg_metrics[2]:.4f}, "
      f"average_profitability_per_trade={avg_metrics[3]:.4f}, "
      f"sharpe_ratio={avg_metrics[4]:.4f}")
