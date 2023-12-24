import os

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
MODEL_INPUT_IN_OBSERVATION = 64
SKIP_STEPS = 1024 + MODEL_INPUT_IN_OBSERVATION

TICKERS = [
    "NEARUSDT", "SOLUSDT", "BTCFDUSD", "BTCUSDT", "ETHUSDT", "MOVRUSDT",
    "AVAXUSDT", "USDCUSDT", "OPUSDT", "DOTUSDT", "FDUSDUSDT"
]


class MultiScaler:
    def __init__(self, min_max: MinMaxScaler, std: StandardScaler):
        self.min_max = min_max
        self.std = std


def preprocess(df: pd.DataFrame, scaler=None):
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
    dataset = add_features(df)
    df_preprocessed, scaler = preprocess(dataset, scaler=scaler)
    return df_preprocessed, scaler


def full_handle_tickers(df_tickers):
    combined_dataset = pd.DataFrame()

    # Combine data from all tickers with added features
    for df_ticker in df_tickers:
        dataset = df_ticker.loc[:, OHLC_COLUMNS].astype(np.float32)
        dataset = add_features(pd.DataFrame(dataset.to_numpy(), columns=OHLC_COLUMNS))
        combined_dataset = pd.concat([combined_dataset, dataset])

    # Preprocess the combined dataset
    combined_preprocessed, combined_scaler = preprocess(combined_dataset)

    results = []

    # Transform each ticker separately using the combined scaler
    for df_ticker in df_tickers:
        dataset = df_ticker.loc[:, OHLC_COLUMNS].astype(np.float32)
        dataset = add_features(pd.DataFrame(dataset.to_numpy(), columns=OHLC_COLUMNS))

        # Use the combined scaler to transform the dataset
        df_scaled, _ = preprocess(dataset, combined_scaler)

        results.append((df_scaled, dataset, combined_scaler))

    return results


def invert_preprocess(original_start, scaler: MultiScaler, df):
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


def add_features(df):
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
    cci = trend.CCIIndicator(df['High'], df['Low'], df['Close'])
    df['CCI'] = cci.cci()

    # Add ADX
    adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()

    # Add Stochastic Oscillator
    indicator_so = momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['stoch'] = indicator_so.stoch()

    # Drop NaN rows resulting from the indicator calculations
    df.dropna(inplace=True)
    return df


def calculate_observation(preprocessed_df, pristine_df, buy_price):
    original_start = pristine_df.iloc[-1]

    curr_close = original_start.Close
    prev_close = pristine_df.iloc[-2].Close

    if buy_price is None:
        current_gain = 0
    else:
        current_gain = ((curr_close - buy_price) / buy_price)
    previous_prices = preprocessed_df.iloc[-MODEL_INPUT_IN_OBSERVATION:].to_numpy()  # shape = (N, 7)
    buy_status = 1 if buy_price is not None else 0
    observation = {
        OBS_PRICES_SEQUENCE: previous_prices.astype(np.float32),
        OBS_OTHER: np.concatenate([[buy_status], [current_gain]]).astype(np.float32)
    }

    return observation, curr_close, prev_close


def test_preprocess_invert_preprocess(original_df):
    from sklearn.metrics import mean_absolute_error
    preprocessed_df, scaler = preprocess(original_df)

    # Assume that 'original_start' is the first row of the original DataFrame
    original_start = original_df.iloc[0]

    inverted_df = invert_preprocess(original_start, scaler, preprocessed_df)

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
    range_preproc, s = preprocess(range_orig)

    # Inverted for the whole preprocessed range
    range1_inv = invert_preprocess(range_orig.iloc[0], s, range_preproc)

    # Inverted for the latter part of the preprocessed range
    range2_inv = invert_preprocess(range_orig.iloc[500], s, range_preproc.iloc[500:])

    # Due to floating point errors, equality may not be exact. So you might use pd.testing.assert_frame_equal
    # with the check_exact=False parameter
    pd.testing.assert_frame_equal(range1_inv.iloc[500:].reset_index(drop=True),
                                  range2_inv.reset_index(drop=True), check_exact=False)


def download_data(refresh=True):
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=".",
        asset_class="spot",  # spot, um, cm
        data_type="klines",  # aggTrades, klines, trades
        data_frequency="1m",
    )

    print(data_dumper.get_list_all_trading_pairs())

    if refresh:
        data_dumper.dump_data(tickers=TICKERS)

    return list(map(lambda ticker: get_df_for_ticker(ticker), TICKERS))


def get_df_for_ticker(ticker):
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
