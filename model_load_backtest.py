import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data
from trading_metrics import max_earning_rate, maximum_pullback, average_profitability_per_trade, sharpe_ratio, \
    cumulative_return, calculate_metrics
from util import download_and_process_data_if_available, SKIP_STEPS

rl_model = PPO.load("cherry-picked-best-models/rl-model-best1.pt")

df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

all_metrics_list = []

for _, df, scaler, name in df_tickers:
    t = time.time()
    print(f"backtest for {name} started")

    bt = create_backtest_model_with_data(rl_model, df, scaler, "2023-5-1", "2023-5-14")
    stats = bt.run()
    equity = bt._results._equity_curve["Equity"].iloc[SKIP_STEPS:]
    y = (equity - 1_000_000) / 1_000_000
    plt.plot(y, label=name)

    print(f"backtest for {name} finished; time taken: {time.time() - t}")

    all_metrics_list.append(calculate_metrics(equity))

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()
plt.legend()
plt.tight_layout()

plt.show()

avg_metrics = np.mean(np.array(all_metrics_list), axis=0)
print(f"Average Metrics: cumulative_return={avg_metrics[0]:.4f}, "
      f"max_earning_rate={avg_metrics[1]:.4f}, "
      f"maximum_pullback={avg_metrics[2]:.4f}, "
      f"average_profitability_per_trade={avg_metrics[3]:.4f}, "
      f"sharpe_ratio={avg_metrics[4]:.4f}")
