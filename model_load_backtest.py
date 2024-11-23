import time

from utils.trading_metrics import calculate_metrics
from utils.util import save_pickle
import pandas as pd


def run_backtest_on_all_tickers(
    df_tickers,
    strat_name,
    in_obs,
    create_strategy_func,
    computed_data_dir=None,
    time_delta_days=7,
):
    sum_equity = None
    trades = 0
    for _, df, scaler, name in df_tickers:
        df = df.read()
        start = df.index.max() - pd.Timedelta(days=time_delta_days)
        end = df.index.max()

        t = time.time()
        print(f"backtest for {name} started")

        skip_steps = 1024 + in_obs
        bt = create_strategy_func(df, scaler, in_obs, start, end)
        res = bt.run()

        if computed_data_dir is not None:
            save_pickle(
                (res._trades, res._equity_curve),
                f"{computed_data_dir}/backtest-results/{strat_name}_{name}.pkl",
            )

        trades += len(res._trades)

        equity = res._equity_curve["Equity"].iloc[skip_steps:]
        if sum_equity is None:
            sum_equity = equity
        else:
            sum_equity += equity

        print(f"backtest for {name} finished; time taken: {time.time() - t}")
    start_cash = 1_000_000

    return strat_name, sum_equity, trades, start_cash


def print_metrics(strat_name, sum_equity, trades, start_cash):
    metrics = calculate_metrics(sum_equity, trades, start_cash)
    print()
    print(f"{strat_name} metrics:")
    print(
        f"cumulative_return={metrics[0]:.4f}, "
        f"max_earning_rate={metrics[1]:.4f}, "
        f"maximum_pullback={metrics[2]:.4f}, "
        f"average_profitability_per_trade={metrics[3]:.4f}, "
        f"sharpe_ratio={metrics[4]:.4f}"
    )
    print()
