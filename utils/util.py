import datetime
import os
import pickle
import re

import line_profiler
import numpy as np
import pandas as pd
from binance_historical_data import BinanceDataDumper
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from ta import trend, momentum

OHLC_COLUMNS = ["Open", "High", "Low", "Close"]

OBS_OTHER = "other"
OBS_PRICES_SEQUENCE = "prices_sequence"

DEFAULT_TICKERS = [
    "NEARUSDT",
    "SOLUSDT",
    "ETHUSDT",
    "BNBUSDT"
]

BINANCE_DATA_START_DATE = datetime.date(2023, 1, 1)


@line_profiler.profile
def preprocess_make_ohlc_relative(df: pd.DataFrame):
    # Apply percentage change only to OHLC columns
    df_pct = df[OHLC_COLUMNS].pct_change()

    # Clamp
    df_pct = np.tanh(df_pct * 250)

    # Concatenate the percentage-changed OHLC with the other columns
    df_all = pd.concat([df_pct, df.drop(columns=OHLC_COLUMNS)], axis=1)

    # Drop NA values (from pct_change operation)
    df_all.dropna(inplace=True)

    return df_all


@line_profiler.profile
def scale_dataframe(df_all: pd.DataFrame, scaler=None):
    if scaler is None:
        scaler = RobustScaler(copy=False)
        df_scaled = scaler.fit_transform(df_all)
    else:
        df_scaled = scaler.transform(df_all)

    df_scaled = pd.DataFrame(
        df_scaled.copy(), columns=df_all.columns, index=df_all.index
    )

    return df_scaled, scaler


@line_profiler.profile
def __full_handle_tickers(df_tickers, sl=1, tp=1):
    results = []

    for i, (df_ticker, ticker_name) in enumerate(df_tickers):
        # Process OHLC data
        dataset = df_ticker.loc[:, OHLC_COLUMNS].astype(np.float32)
        dataset.index = pd.to_datetime(df_ticker["Open time"], unit="us")

        # Add features
        dataset_with_features = preprocess_add_features(
            pd.DataFrame(dataset, columns=OHLC_COLUMNS)
        )

        # Make relative
        preprocessed_dataset = preprocess_make_ohlc_relative(dataset_with_features)

        # Scale individually
        df_scaled, individual_scaler = scale_dataframe(preprocessed_dataset)

        # Generate labels
        labels = generate_labels_for_supervised(dataset_with_features.iloc[1:], sl, tp)

        # Store results
        results.append((df_scaled, dataset_with_features, labels, individual_scaler, ticker_name))

    return results


@line_profiler.profile
def generate_labels_for_supervised(pristine, sl, tp):
    lookahead_steps = 256

    close_prices = pristine['Close'].to_numpy()
    ticker_labels = np.zeros(len(close_prices), dtype=int)

    for i in range(len(close_prices) - 1):
        current_price = close_prices[i]
        sl_price = stop_loss_price(current_price, sl)
        tp_price = take_profit_price(current_price, tp)

        # Limit future prices to the next 256 timesteps
        future_prices = close_prices[i + 1:i + 1 + lookahead_steps]

        sl_triggered = np.sum(future_prices <= sl_price)
        tp_triggered = np.sum(future_prices >= tp_price)

        if tp_triggered > sl_triggered:
            ticker_labels[i] = 1

    return pd.DataFrame(ticker_labels, index=pristine.index, columns=['Label'])


def split_tickers_train_test(df_tickers, last_days):
    last_date = df_tickers[0][0].index.max() - pd.Timedelta(days=last_days)

    df_tickers_train = list(
        map(
            lambda ticker: (
                ticker[0].loc[:last_date],
                ticker[1].loc[:last_date],
                ticker[2].loc[:last_date],
                ticker[3],
                ticker[4],
            ),
            df_tickers,
        )
    )

    df_tickers_test = list(
        map(
            lambda ticker: (
                ticker[0].loc[(last_date + pd.Timedelta(seconds=1)):],  # Add 1 second to exclude last_date
                ticker[1].loc[(last_date + pd.Timedelta(seconds=1)):],
                ticker[2].loc[(last_date + pd.Timedelta(seconds=1)):],
                ticker[3],
                ticker[4],
            ),
            df_tickers,
        )
    )

    return df_tickers_train, df_tickers_test


def __invert_preprocess(original_start, scaler: RobustScaler, df):
    df = df.copy()

    original_start = original_start[OHLC_COLUMNS].to_numpy()
    # Invert MinMax scaling for all columns
    df_inv_scaled = pd.DataFrame(
        scaler.inverse_transform(df.to_numpy()),
        columns=df.columns,
        index=df.index,
    )

    # Recover the original OHLC values
    reversed_array = np.cumprod(1 + df_inv_scaled[OHLC_COLUMNS].to_numpy(), axis=0)
    # Scaling by the original_start to each element
    reversed_array = reversed_array * original_start

    df_inv_scaled[OHLC_COLUMNS] = reversed_array

    return df_inv_scaled


def preprocess_add_features(df):
    df = df.copy()
    # Add Simple Moving Averages (SMAs)
    df["SMA_256"] = df["Close"].rolling(window=256).mean()
    df["SMA_512"] = df["Close"].rolling(window=512).mean()
    df["SMA_1024"] = df["Close"].rolling(window=1024).mean()

    # Convert SMA columns to distance in percentages from "Close"
    df["SMA_256"] = np.tanh((df["Close"] - df["SMA_256"]) / df["SMA_256"] * 25)
    df["SMA_512"] = np.tanh((df["Close"] - df["SMA_512"]) / df["SMA_512"] * 25)
    df["SMA_1024"] = np.tanh((df["Close"] - df["SMA_1024"]) / df["SMA_1024"] * 25)

    # Add MACD
    macd = trend.MACD(df["Close"])
    df["MACD_diff"] = np.tanh((macd.macd_diff() * 600 / df["Close"]))

    # Add RSI
    df["RSI"] = momentum.RSIIndicator(df["Close"]).rsi()

    # Add CCI
    # cci = trend.CCIIndicator(df['High'], df['Low'], df['Close'])
    # df['CCI'] = cci.cci()

    # Add ADX
    # adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    # df['ADX'] = adx.adx()

    # Add Stochastic Oscillator
    indicator_so = momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"]
    )
    df["stoch"] = indicator_so.stoch()

    # Drop NaN rows resulting from the indicator calculations
    df.dropna(inplace=True)
    return df


def __download_data(data_dir, need_download, tickers):
    if need_download:
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump=f"{data_dir}/",
            asset_class="spot",  # spot, um, cm
            data_type="klines",  # aggTrades, klines, trades
            data_frequency="1m",
        )

        print(data_dumper.get_list_all_trading_pairs())

        data_dumper.dump_data(tickers=tickers, date_start=BINANCE_DATA_START_DATE, is_to_update_existing=True)

    return list(
        zip(map(lambda ticker: __get_df_for_ticker(data_dir, ticker), tickers), tickers)
    )


def __get_df_for_ticker(data_dir, ticker):
    minute_klines_dir = f"{data_dir}/spot/monthly/klines/{ticker}/1m"
    filenames = next(os.walk(minute_klines_dir), (None, None, []))[2]  # [] if no file

    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]

    df = pd.DataFrame(columns=columns)

    for f in filenames:
        new_df = pd.read_csv(f"{minute_klines_dir}/{f}", header=None, names=columns)
        df = pd.concat([d for d in [df, new_df] if not d.empty], ignore_index=True)
    df = df.sort_values(by="Open time")
    return df


def save_pickle(data, filename):
    create_dir_if_not_exists(filename)

    # Save the processed data to disk
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def create_dir_if_not_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_pickle(filename):
    # Load the processed data from disk
    with open(filename, "rb") as file:
        return pickle.load(file)


def download_and_process_data_if_available(data_dir, reload=False, need_download=True, sl=1, tp=1, tickers=None):
    # Check if the processed data already exists
    if tickers is None:
        tickers = DEFAULT_TICKERS

    cache_path = f"{data_dir}/df_tickers.pkl"
    if os.path.exists(cache_path) and not reload:
        print("Loading data from cache")
        return load_pickle(cache_path)
    else:
        print("Downloading and processing data")
        df_tickers = __download_data(data_dir, need_download, tickers)
        df_tickers_processed = __full_handle_tickers(df_tickers, sl, tp)
        save_pickle(df_tickers_processed, cache_path)
        return df_tickers_processed


def create_synthetic_price_data():
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    def generate_synthetic_data(tickers, start_date, end_date, freq="1T"):
        data = []
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        time_steps = np.arange(len(date_range))

        for ticker in tickers:
            # Sinusoidal prices with random noise and shift
            shift = np.random.uniform(-np.pi, np.pi)
            base_prices = 50 + 10 * np.sin((time_steps / 180) + shift)
            prices = base_prices

            volumes = np.random.randint(100, 10000, size=len(date_range))

            df = pd.DataFrame(
                {
                    "Open time": date_range,
                    "Open": prices,
                    "High": prices,
                    "Low": prices,
                    "Close": prices,
                    "Volume": volumes,
                }
            )
            data.append((df, ticker))
        return data

    print("Creating synthetic data")
    tickers = ["SYNTH1USDT"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    df_tickers = generate_synthetic_data(tickers, start_date, end_date)
    df_tickers_processed = __full_handle_tickers(df_tickers)
    return df_tickers_processed


def create_random_walk_price_data():
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    def generate_random_walk_data(
            tickers, start_date, end_date, initial_price=100, freq="1T", volatility=0.02
    ):
        data = []
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        num_periods = len(date_range)

        for ticker in tickers:
            prices = [initial_price]
            for _ in range(1, num_periods):
                change_percent = np.random.uniform(-volatility, volatility)
                new_price = prices[-1] * (1 + change_percent)
                prices.append(new_price)

            volumes = np.random.randint(100, 10000, size=num_periods)

            df = pd.DataFrame(
                {
                    "Open time": date_range,
                    "Open": prices,
                    "High": prices,
                    "Low": prices,
                    "Close": prices,
                    "Volume": volumes,
                }
            )
            data.append((df, ticker))
        return data

    print("Creating random walk price data")
    tickers = ["RANDOM1USDT"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    df_tickers = generate_random_walk_data(tickers, start_date, end_date)
    df_tickers_processed = __full_handle_tickers(df_tickers)
    return df_tickers_processed


def get_name_max_timesteps(models_dir):
    files = next(os.walk(models_dir), (None, None, []))[2]
    if len(files) == 0:
        return None

    return max(
        files,
        key=lambda x: int(re.match(".*?(\\d+)_steps\\.zip", x).group(1)),
    )


def stop_loss_price(price, percent):
    return price * (100 - percent) / 100


def take_profit_price(price, percent):
    return price * (100 + percent) / 100
