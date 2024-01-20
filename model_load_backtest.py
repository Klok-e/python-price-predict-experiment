import os
import time

from stable_baselines3 import PPO

from backtest import create_backtest_model_with_data, create_buy_and_hold_strategy
from trading_metrics import calculate_metrics
from util import download_and_process_data_if_available, save_pickle


def run_backtest_on_all_tickers(strat_name, in_obs, create_strategy_func):
    sum_equity = None
    trades = 0
    for _, df, scaler, name in df_tickers:
        start = "2023-07-15"
        end = "2023-09-01"

        t = time.time()
        print(f"backtest for {name} started")

        skip_steps = 1024 + in_obs
        bt = create_strategy_func(df, scaler, in_obs, start, end)
        res = bt.run()

        save_pickle((res._trades, res._equity_curve), f"backtest-results/{strat_name}_{name}.pkl")

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


df_tickers = download_and_process_data_if_available("cache/df_tickers.pkl")

# computed_data_dir = "/content/drive/MyDrive/colab/computed-data"
computed_data_dir = "./rl-drom-google-drive"
dirnames = next(os.walk(f"{computed_data_dir}/rl-model/"), (None, [], None))[1]  # [] if no file
for i, dir in enumerate(sorted(dirnames)):
    model_path = f"{computed_data_dir}/rl-model/{dir}/checkpoints/{dir}_2499750_steps.zip"
    print(model_path)

    window_size = int(dir.split("_")[3].replace("ws", ""))
    print(window_size)

    rl_model = PPO.load(model_path)
    run_backtest_on_all_tickers(f"Model {dir}",
                                window_size,
                                lambda df, scaler, model_in_observations, start,
                                       end: create_backtest_model_with_data(
                                    rl_model,
                                    df,
                                    scaler,
                                    start,
                                    end,
                                    model_in_observations))

run_backtest_on_all_tickers("Buy and Hold",
                            64,
                            lambda df, scaler, model_in_observations, start,
                                   end: create_buy_and_hold_strategy(
                                df,
                                start,
                                end))
