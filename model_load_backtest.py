import time

from trading_metrics import calculate_metrics
from util import save_pickle


def run_backtest_on_all_tickers(df_tickers, strat_name, in_obs, create_strategy_func, computed_data_dir):
    sum_equity = None
    trades = 0
    for _, df, scaler, name in df_tickers:
        start = "2024-02-01"
        end = "2024-03-01"

        t = time.time()
        print(f"backtest for {name} started")

        skip_steps = 1024 + in_obs
        bt = create_strategy_func(df, scaler, in_obs, start, end)
        res = bt.run()

        save_pickle((res._trades, res._equity_curve), f"{computed_data_dir}/backtest-results/{strat_name}_{name}.pkl")

        trades += len(res._trades)

        equity = res._equity_curve["Equity"].iloc[skip_steps:]
        if sum_equity is None:
            sum_equity = equity
        else:
            sum_equity += equity

        print(f"backtest for {name} finished; time taken: {time.time() - t}")
    start_cash = 1_000_000

    metrics = calculate_metrics(sum_equity, trades, start_cash)
    print()
    print(f"{strat_name} metrics:")
    print(f"cumulative_return={metrics[0]:.4f}, "
          f"max_earning_rate={metrics[1]:.4f}, "
          f"maximum_pullback={metrics[2]:.4f}, "
          f"average_profitability_per_trade={metrics[3]:.4f}, "
          f"sharpe_ratio={metrics[4]:.4f}")
    print()
