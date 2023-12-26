import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data
from util import download_and_process_data_if_available

rl_model = PPO.load("cherry-picked-best-models/rl-model-best1.pt")

df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

for _, df, scaler in df_tickers[:2]:
    bt = create_backtest_model_with_data(rl_model, df, scaler, "2023-5-1", "2023-5-4")
    stats = bt.run()
    plt.plot(bt._results._equity_curve["Equity"])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

# Improve layout
plt.tight_layout()

plt.show()