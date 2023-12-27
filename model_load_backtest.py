import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import dates as mdates
from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data
from trading_metrics import max_earning_rate, maximum_pullback, average_profitability_per_trade, sharpe_ratio, \
    cumulative_return
from util import download_and_process_data_if_available, SKIP_STEPS

rl_model = PPO.load("cherry-picked-best-models/rl-model-best1.pt")

df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

for _, df, scaler, name in df_tickers:
    t = time.time()
    print(f"backtest for {name} started")
    bt = create_backtest_model_with_data(rl_model, df, scaler, "2023-5-1", "2023-5-14")
    stats = bt.run()
    equity = bt._results._equity_curve["Equity"].iloc[SKIP_STEPS:]
    y = (equity - 1_000_000) / 1_000_000
    plt.plot(y, label=name)
    print(f"backtest for {name} finished; time taken: {time.time() - t}")

    # Calculate metrics
    P_end = equity.iloc[-1]
    P_0 = equity.iloc[0]
    NT = len(equity)  # Assuming one trade per time step
    E_Rp = equity.pct_change().mean()  # Expected return
    sigma_P = equity.pct_change().std()  # Standard deviation of returns

    # You'll need to adapt the below calculations depending on how your 'stats' are structured
    A_x = equity.values  # Assuming these are total assets over time
    A_y = np.roll(A_x, 1)  # Shifted assets for comparison
    A_y[0] = 1_000_000  # Initial asset value for the first calculation

    # Assuming 252 trading days in a year and 6.5 trading hours per day
    minutes_in_trading_year = 252 * 6.5 * 60
    # Your risk-free rate, annualized
    annual_risk_free_rate = 0.02
    # Convert the annual risk-free rate to a per-minute rate
    R_f = (1 + annual_risk_free_rate) ** (1 / minutes_in_trading_year) - 1

    # Annualize the expected return and standard deviation
    annualized_E_Rp = E_Rp * np.sqrt(minutes_in_trading_year)
    annualized_sigma_P = sigma_P * np.sqrt(minutes_in_trading_year)

    cr = cumulative_return(P_end, P_0)
    mer = max_earning_rate(A_x, A_y)
    mpb = maximum_pullback(A_x, A_y)
    appt = average_profitability_per_trade(P_end, P_0, NT)
    sr = sharpe_ratio(E_Rp, R_f, sigma_P)

    print(f"Metrics for {name}: CR={cr:.4f}, MER={mer:.4f}, MPB={mpb:.4f}, APPT={appt:.4f}, SR={sr:.4f}")

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

plt.legend()

# Improve layout
plt.tight_layout()

plt.show()
