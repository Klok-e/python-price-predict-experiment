import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data
from util import download_and_process_data_if_available

rl_model = PPO.load("cherry-picked-best-models/rl-model-best1.pt")

df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

for _, df, scaler in df_tickers:
    bt = create_backtest_model_with_data(rl_model, df, scaler, -5000, 0)
    stats = bt.run()
    plt.plot(bt._results._equity_curve["Equity"].index, bt._results._equity_curve["Equity"])

plt.show()
