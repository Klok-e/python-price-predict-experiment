import datetime
import os
import pickle

import numpy as np
import pandas as pd
from binance_historical_data import BinanceDataDumper
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta import trend, momentum

OHLC_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close"]

OBS_OTHER = "other"
OBS_PRICES_SEQUENCE = "prices_sequence"

TICKERS = [
    "NEARUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"
]


class MultiScaler:
    def __init__(self, min_max: MinMaxScaler, std: StandardScaler):
        self.min_max = min_max
        self.std = std


# @profile
def preprocess_scale(df: pd.DataFrame, scaler=None):
    # Apply percentage change only to OHLC columns
    df_pct = df[OHLC_COLUMNS].pct_change()

    # Clamp
    df_pct = np.tanh(df_pct)

    # Concatenate the percentage-changed OHLC with the other columns
    df_all = pd.concat([df_pct, df.drop(columns=OHLC_COLUMNS)], axis=1)

    # Drop NA values (from pct_change operation)
    df_all.dropna(inplace=True)

    # Apply MinMax scaling to all columns
    if scaler is None:
        scaler = MultiScaler(MinMaxScaler(feature_range=(-1, 1), copy=False), StandardScaler(copy=False))
        df_multi_scaled = scaler.min_max.fit_transform(scaler.std.fit_transform(df_all))
    else:
        df_multi_scaled = scaler.min_max.transform(scaler.std.transform(df_all))

    df_scaled = pd.DataFrame(df_multi_scaled.copy(),
                             columns=df_all.columns,
                             index=df_all.index)

    return df_scaled, scaler


def full_preprocess(df: pd.DataFrame, scaler=None):
    df_features = preprocess_add_features(df)
    df_preprocessed, scaler = preprocess_scale(df_features, scaler=scaler)
    return df_preprocessed, scaler


def __full_handle_tickers(df_tickers):
    datasets = []

    for df_ticker, _ in df_tickers:
        dataset = df_ticker.loc[:, OHLC_COLUMNS].astype(np.float32)
        dataset.index = pd.to_datetime(df_ticker["Open time"], unit='ms')
        dataset_with_features = preprocess_add_features(pd.DataFrame(dataset, columns=OHLC_COLUMNS))
        datasets.append(dataset_with_features)

    combined_dataset = pd.concat(datasets)

    combined_preprocessed, combined_scaler = preprocess_scale(combined_dataset)

    results = []

    for i, dataset in enumerate(datasets):
        df_scaled, _ = preprocess_scale(dataset, combined_scaler)
        results.append((df_scaled, dataset.iloc[1:], combined_scaler, df_tickers[i][1]))

    return results


def __invert_preprocess(original_start, scaler: MultiScaler, df):
    df = df.copy()

    original_start = original_start[OHLC_COLUMNS].to_numpy()
    # Invert MinMax scaling for all columns
    df_inv_scaled = pd.DataFrame(scaler.std.inverse_transform(scaler.min_max.inverse_transform(df.to_numpy())),
                                 columns=df.columns,
                                 index=df.index)

    # Recover the original OHLC values
    reversed_array = np.cumprod(1 + df_inv_scaled[OHLC_COLUMNS].to_numpy(), axis=0)
    reversed_array = reversed_array * original_start  # Scaling by the original_start to each element

    df_inv_scaled[OHLC_COLUMNS] = reversed_array

    return df_inv_scaled


def preprocess_add_features(df):
    df = df.copy()
    # Add Simple Moving Averages (SMAs)
    df['SMA_256'] = df['Close'].rolling(window=256).mean()
    df['SMA_512'] = df['Close'].rolling(window=512).mean()
    df['SMA_1024'] = df['Close'].rolling(window=1024).mean()

    # Convert SMA columns to distance in percentages from "Close"
    df['SMA_256'] = ((df['Close'] - df['SMA_256']) / df['SMA_256'])
    df['SMA_512'] = ((df['Close'] - df['SMA_512']) / df['SMA_512'])
    df['SMA_1024'] = ((df['Close'] - df['SMA_1024']) / df['SMA_1024'])

    # Add MACD
    macd = trend.MACD(df['Close'])
    df['MACD_diff'] = macd.macd_diff()

    # Add RSI
    df['RSI'] = momentum.RSIIndicator(df['Close']).rsi()

    # Add CCI
    # cci = trend.CCIIndicator(df['High'], df['Low'], df['Close'])
    # df['CCI'] = cci.cci()

    # Add ADX
    # adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    # df['ADX'] = adx.adx()

    # Add Stochastic Oscillator
    indicator_so = momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['stoch'] = indicator_so.stoch()

    # Drop NaN rows resulting from the indicator calculations
    df.dropna(inplace=True)
    return df


# @profile
def calculate_observation(preprocessed_df, pristine_df, buy_price):
    curr_close = pristine_df.iloc[-1].Close
    prev_close = pristine_df.iloc[-2].Close

    if buy_price is None:
        current_gain = 0
    else:
        current_gain = ((curr_close - buy_price) / buy_price)
    previous_prices = preprocessed_df.to_numpy()  # shape = (N, 7)
    buy_status = 1 if buy_price is not None else 0
    observation = {
        OBS_PRICES_SEQUENCE: previous_prices.astype(np.float32),
        OBS_OTHER: np.concatenate([[buy_status], [current_gain]]).astype(np.float32)
    }

    return observation, curr_close, prev_close


def test_preprocess_invert_preprocess(original_df):
    from sklearn.metrics import mean_absolute_error
    preprocessed_df, scaler = preprocess_scale(original_df)

    # Assume that 'original_start' is the first row of the original DataFrame
    original_start = original_df.iloc[0]

    inverted_df = __invert_preprocess(original_start, scaler, preprocessed_df)

    mae_list = []
    for col in original_df.columns:
        # Start from the second row of the original_df for comparison
        mae = mean_absolute_error(original_df.iloc[1:][col], inverted_df[col])
        mae_list.append(mae)
        print(f"Mean Absolute Error for {col}: {mae}")

    avg_mae = sum(mae_list) / len(mae_list)
    print(f"Average MAE: {avg_mae}")

    return avg_mae < 1e-9


def test_orig_val(dataset):
    # The original_start passed to invert_preprocess() must be the first value in the corresponding
    # original DataFrame segment. For the first range, that's range_orig.iloc[0].
    # For the second range, it's range_orig.iloc[500].

    range_orig = dataset.iloc[2000:3000]
    range_preproc, s = preprocess_scale(range_orig)

    # Inverted for the whole preprocessed range
    range1_inv = __invert_preprocess(range_orig.iloc[0], s, range_preproc)

    # Inverted for the latter part of the preprocessed range
    range2_inv = __invert_preprocess(range_orig.iloc[500], s, range_preproc.iloc[500:])

    # Due to floating point errors, equality may not be exact. So you might use pd.testing.assert_frame_equal
    # with the check_exact=False parameter
    pd.testing.assert_frame_equal(range1_inv.iloc[500:].reset_index(drop=True),
                                  range2_inv.reset_index(drop=True), check_exact=False)


def __download_data(refresh=True):
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=".",
        asset_class="spot",  # spot, um, cm
        data_type="klines",  # aggTrades, klines, trades
        data_frequency="1m",
    )

    print(data_dumper.get_list_all_trading_pairs())

    if refresh:
        data_dumper.dump_data(tickers=TICKERS, date_start=datetime.date(2019, 1, 1))

    return list(zip(map(lambda ticker: __get_df_for_ticker(ticker), TICKERS), TICKERS))


def __get_df_for_ticker(ticker):
    filenames = next(os.walk(f"./spot/monthly/klines/{ticker}/1m"), (None, None, []))[2]  # [] if no file

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
        "Ignore"
    ]

    df = pd.DataFrame(columns=columns)

    for f in filenames:
        new_df = pd.read_csv(f"./spot/monthly/klines/{ticker}/1m/{f}", header=None, names=columns)
        df = pd.concat([d for d in [df, new_df] if not d.empty], ignore_index=True)
    df = df.sort_values(by="Open time")
    return df


def save_pickle(data, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Save the processed data to disk
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(filename):
    # Load the processed data from disk
    with open(filename, 'rb') as file:
        return pickle.load(file)


def download_and_process_data_if_available(filename):
    # Check if the processed data already exists
    if os.path.exists(filename):
        print("Loading data from cache")
        return load_pickle(filename)
    else:
        print("Downloading and processing data")
        df_tickers = __download_data()
        df_tickers_processed = __full_handle_tickers(df_tickers)
        save_pickle(df_tickers_processed, filename)
        return df_tickers_processed


def create_synthetic_price_data():
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    def generate_synthetic_data(tickers, start_date, end_date, freq='1T'):
        data = []
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        time_steps = np.arange(len(date_range))

        for ticker in tickers:
            # Sinusoidal prices with random noise and shift
            shift = np.random.uniform(-np.pi, np.pi)  # Random shift
            base_prices = 50 + 10 * np.sin((time_steps / 180) + shift)  # Base sinusoidal function with shift
            # noise = np.random.normal(0, 2, size=len(time_steps))  # Random noise
            prices = base_prices #+ noise

            volumes = np.random.randint(100, 10000, size=len(date_range))

            df = pd.DataFrame({
                'Open time': date_range,
                'Open': prices,
                'High': prices * np.random.uniform(1.01, 1.05, size=len(date_range)),
                'Low': prices * np.random.uniform(0.95, 0.99, size=len(date_range)),
                'Close': prices,
                'Volume': volumes
            })
            data.append((df, ticker))
        return data

    print("Creating synthetic data")
    tickers = ["SYNTH1USDT"]#, "SYNTH2USDT"]
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    df_tickers = generate_synthetic_data(tickers, start_date, end_date)
    df_tickers_processed = __full_handle_tickers(df_tickers)
    return df_tickers_processed
