import datetime
import os
import pickle
import re

import numpy as np
import pandas as pd
try:
    import line_profiler
except ModuleNotFoundError:
    class _LineProfilerFallback:
        @staticmethod
        def profile(func):
            return func

    line_profiler = _LineProfilerFallback()

try:
    from binance_historical_data import BinanceDataDumper
except ModuleNotFoundError:
    BinanceDataDumper = None

from sklearn.preprocessing import RobustScaler
from ta import trend, momentum
from utils.experiment import DEFAULT_TRADING_CONTRACT, TradingContract, stable_config_hash

OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
FEATURE_COLUMNS = ["SMA_256", "SMA_512", "SMA_1024", "MACD_diff", "RSI", "stoch"]

OBS_OTHER = "other"
OBS_PRICES_SEQUENCE = "prices_sequence"

DEFAULT_TICKERS = [
    "NEARUSDT",
    "SOLUSDT",
    "ETHUSDT",
    "BNBUSDT"
]

BINANCE_DATA_START_DATE = datetime.date(2023, 1, 1)


def _contract_from_args(sl=None, tp=None, contract: TradingContract | None = None):
    if contract is not None:
        return contract

    if sl is None:
        sl = DEFAULT_TRADING_CONTRACT.stop_loss_percent
    if tp is None:
        tp = DEFAULT_TRADING_CONTRACT.take_profit_percent

    return TradingContract(
        stop_loss_percent=sl,
        take_profit_percent=tp,
        lookahead_steps=DEFAULT_TRADING_CONTRACT.lookahead_steps,
        commission=DEFAULT_TRADING_CONTRACT.commission,
    )


def _open_time_to_datetime(open_time):
    if pd.api.types.is_datetime64_any_dtype(open_time):
        return pd.to_datetime(open_time)

    numeric_open_time = pd.to_numeric(open_time, errors="coerce")
    median_open_time = numeric_open_time.dropna().median()

    if median_open_time > 1e17:
        unit = "ns"
    elif median_open_time > 1e14:
        unit = "us"
    elif median_open_time > 1e11:
        unit = "ms"
    else:
        unit = "s"

    return pd.to_datetime(numeric_open_time, unit=unit)


def validate_ohlc_bars(df: pd.DataFrame, ticker_name: str | None = None, frequency="1min"):
    missing_columns = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{ticker_name or 'ticker'} is missing OHLC columns: {missing_columns}")

    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        if "Open time" not in result.columns:
            raise ValueError(f"{ticker_name or 'ticker'} needs a DatetimeIndex or an Open time column")
        result.index = _open_time_to_datetime(result["Open time"])

    result[OHLC_COLUMNS] = result[OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique()
        raise ValueError(f"{ticker_name or 'ticker'} has duplicate bars, first duplicate: {duplicates[0]}")

    if not result.index.is_monotonic_increasing:
        raise ValueError(f"{ticker_name or 'ticker'} has a non-monotonic index")

    if result[OHLC_COLUMNS].isna().any().any():
        raise ValueError(f"{ticker_name or 'ticker'} has NaN OHLC values")

    invalid_ohlc = (
        (result["High"] < result[["Open", "Close", "Low"]].max(axis=1))
        | (result["Low"] > result[["Open", "Close", "High"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError(f"{ticker_name or 'ticker'} has invalid OHLC values at {invalid_ohlc.idxmax()}")

    if frequency is not None and len(result.index) > 1:
        expected_delta = pd.Timedelta(frequency)
        deltas = result.index.to_series().diff().dropna()
        missing_or_irregular = deltas[deltas != expected_delta]
        if not missing_or_irregular.empty:
            raise ValueError(
                f"{ticker_name or 'ticker'} has missing or irregular bars before {missing_or_irregular.index[0]}"
            )

    return result


def ensure_feature_columns(df: pd.DataFrame):
    if all(column in df.columns for column in FEATURE_COLUMNS):
        return df.copy()
    return preprocess_add_features(pd.DataFrame(df.loc[:, OHLC_COLUMNS], columns=OHLC_COLUMNS))


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
def __full_handle_tickers(df_tickers, sl=None, tp=None, contract: TradingContract | None = None):
    contract = _contract_from_args(sl=sl, tp=tp, contract=contract)
    results = []

    for i, (df_ticker, ticker_name) in enumerate(df_tickers):
        dataset = validate_ohlc_bars(df_ticker, ticker_name=ticker_name)
        dataset = dataset.loc[:, OHLC_COLUMNS].astype(np.float32)

        # Add features
        dataset_with_features = preprocess_add_features(pd.DataFrame(dataset, columns=OHLC_COLUMNS))

        # Make relative
        preprocessed_dataset = preprocess_make_ohlc_relative(dataset_with_features)

        # Scale individually
        df_scaled, individual_scaler = scale_dataframe(preprocessed_dataset)

        # Generate labels
        labels = generate_labels_for_supervised(dataset_with_features, contract=contract)
        common_index = preprocessed_dataset.index.intersection(labels.index)
        df_scaled = df_scaled.loc[common_index]
        dataset_with_features = dataset_with_features.loc[common_index]
        labels = labels.loc[common_index]

        # Store results
        results.append((df_scaled, dataset_with_features, labels, individual_scaler, ticker_name))

    return results


@line_profiler.profile
def generate_labels_for_supervised(
        pristine,
        sl=None,
        tp=None,
        contract: TradingContract | None = None,
        drop_incomplete=True,
):
    contract = _contract_from_args(sl=sl, tp=tp, contract=contract)

    if contract.same_candle_policy != "negative":
        raise ValueError("Only same_candle_policy='negative' is currently supported")
    if contract.entry_timing != "next_open":
        raise ValueError("Only entry_timing='next_open' is currently supported")

    required_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [column for column in required_columns if column not in pristine.columns]
    if missing_columns:
        raise ValueError(f"Cannot generate labels without columns: {missing_columns}")

    open_prices = pristine["Open"].to_numpy()
    high_prices = pristine["High"].to_numpy()
    low_prices = pristine["Low"].to_numpy()
    labels = []
    indexes = []

    for signal_idx in range(len(pristine)):
        entry_idx = signal_idx + 1
        horizon_end = entry_idx + contract.lookahead_steps
        if entry_idx >= len(pristine) or horizon_end > len(pristine):
            if not drop_incomplete:
                labels.append(0)
                indexes.append(pristine.index[signal_idx])
            continue

        entry_price = open_prices[entry_idx]
        sl_price = stop_loss_price(entry_price, contract.stop_loss_percent)
        tp_price = take_profit_price(entry_price, contract.take_profit_percent)
        label = 0

        for future_idx in range(entry_idx, horizon_end):
            sl_hit = low_prices[future_idx] <= sl_price
            tp_hit = high_prices[future_idx] >= tp_price

            if sl_hit and tp_hit:
                label = 0
                break
            if tp_hit:
                label = 1
                break
            if sl_hit:
                label = 0
                break

        labels.append(label)
        indexes.append(pristine.index[signal_idx])

    return pd.DataFrame(labels, index=indexes, columns=["Label"], dtype=np.float32)


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


def _process_supervised_segment(
        features: pd.DataFrame,
        contract: TradingContract,
        scaler: RobustScaler | None = None,
        fit_scaler=False,
):
    if features.empty:
        return features.copy(), features.copy(), pd.DataFrame(columns=["Label"], dtype=np.float32), scaler

    labels = generate_labels_for_supervised(features, contract=contract)
    relative_features = preprocess_make_ohlc_relative(features)
    common_index = relative_features.index.intersection(labels.index)
    relative_features = relative_features.loc[common_index]
    labels = labels.loc[common_index]
    original_features = features.loc[common_index]

    if relative_features.empty:
        return relative_features, original_features, labels, scaler

    if fit_scaler:
        scaled_features, scaler = scale_dataframe(relative_features)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler is required for non-training segments")
        scaled_features, _ = scale_dataframe(relative_features, scaler)

    return scaled_features, original_features, labels, scaler


def _split_features_by_days(features: pd.DataFrame, validation_days: int, test_days: int):
    max_date = features.index.max()
    test_start = max_date - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)

    train = features.loc[features.index < validation_start]
    validation = features.loc[(features.index >= validation_start) & (features.index < test_start)]
    test = features.loc[features.index >= test_start]
    return train, validation, test


def prepare_supervised_splits(
        df_tickers,
        contract: TradingContract | None = None,
        validation_days: int = 7,
        test_days: int = 7,
):
    contract = contract or DEFAULT_TRADING_CONTRACT
    train_results = []
    validation_results = []
    test_results = []

    for df_ticker, ticker_name in df_tickers:
        dataset = validate_ohlc_bars(df_ticker, ticker_name=ticker_name)
        dataset = dataset.loc[:, OHLC_COLUMNS].astype(np.float32)
        features = preprocess_add_features(pd.DataFrame(dataset, columns=OHLC_COLUMNS))

        train_features, validation_features, test_features = _split_features_by_days(
            features, validation_days=validation_days, test_days=test_days
        )

        train_scaled, train_original, train_labels, scaler = _process_supervised_segment(
            train_features, contract=contract, fit_scaler=True
        )
        if scaler is None:
            raise ValueError(
                f"{ticker_name} does not have enough training rows for features and "
                f"{contract.lookahead_steps} lookahead steps"
            )
        validation_scaled, validation_original, validation_labels, _ = _process_supervised_segment(
            validation_features, contract=contract, scaler=scaler
        )
        test_scaled, test_original, test_labels, _ = _process_supervised_segment(
            test_features, contract=contract, scaler=scaler
        )

        train_results.append((train_scaled, train_original, train_labels, scaler, ticker_name))
        validation_results.append((validation_scaled, validation_original, validation_labels, scaler, ticker_name))
        test_results.append((test_scaled, test_original, test_labels, scaler, ticker_name))

    return train_results, validation_results, test_results


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
        if BinanceDataDumper is None:
            raise RuntimeError("binance_historical_data is required for downloads")

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
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def load_pickle(filename):
    # Load the processed data from disk
    with open(filename, "rb") as file:
        return pickle.load(file)


def download_and_process_data_if_available(
        data_dir,
        reload=False,
        need_download=True,
        sl=None,
        tp=None,
        tickers=None,
        contract: TradingContract | None = None,
):
    # Check if the processed data already exists
    if tickers is None:
        tickers = DEFAULT_TICKERS

    contract = _contract_from_args(sl=sl, tp=tp, contract=contract)
    cache_key = stable_config_hash({
        "pipeline": "full_processed_v2",
        "tickers": tuple(tickers),
        "contract": contract,
    })
    cache_path = f"{data_dir}/df_tickers_{cache_key}.pkl"
    if os.path.exists(cache_path) and not reload:
        print("Loading data from cache")
        return load_pickle(cache_path)
    else:
        print("Downloading and processing data")
        df_tickers = __download_data(data_dir, need_download, tickers)
        df_tickers_processed = __full_handle_tickers(df_tickers, contract=contract)
        save_pickle(df_tickers_processed, cache_path)
        return df_tickers_processed


def download_and_prepare_splits_if_available(
        data_dir,
        reload=False,
        need_download=True,
        tickers=None,
        contract: TradingContract | None = None,
        validation_days: int = 7,
        test_days: int = 7,
):
    if tickers is None:
        tickers = DEFAULT_TICKERS

    contract = contract or DEFAULT_TRADING_CONTRACT
    cache_key = stable_config_hash({
        "pipeline": "strict_splits_v1",
        "tickers": tuple(tickers),
        "contract": contract,
        "validation_days": validation_days,
        "test_days": test_days,
    })
    cache_path = f"{data_dir}/df_ticker_splits_{cache_key}.pkl"

    if os.path.exists(cache_path) and not reload:
        print("Loading split data from cache")
        return load_pickle(cache_path)

    print("Downloading and preparing split data")
    df_tickers = __download_data(data_dir, need_download, tickers)
    splits = prepare_supervised_splits(
        df_tickers,
        contract=contract,
        validation_days=validation_days,
        test_days=test_days,
    )
    save_pickle(splits, cache_path)
    return splits


def create_synthetic_ohlc_data(days=14, tickers=None, freq="1min"):
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

    if tickers is None:
        tickers = ["SYNTH1USDT"]

    end_date = datetime.now().replace(second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    return generate_synthetic_data(tickers, start_date, end_date, freq=freq)


def create_synthetic_price_data(days=365, contract: TradingContract | None = None):
    print("Creating synthetic data")
    df_tickers = create_synthetic_ohlc_data(days=days)
    df_tickers_processed = __full_handle_tickers(df_tickers, contract=contract)
    return df_tickers_processed


def create_random_walk_ohlc_data(days=14, tickers=None, freq="1min", initial_price=100, volatility=0.02):
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

    if tickers is None:
        tickers = ["RANDOM1USDT"]

    end_date = datetime.now().replace(second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    return generate_random_walk_data(
        tickers,
        start_date,
        end_date,
        initial_price=initial_price,
        freq=freq,
        volatility=volatility,
    )


def create_random_walk_price_data(days=365, contract: TradingContract | None = None):
    print("Creating random walk price data")
    df_tickers = create_random_walk_ohlc_data(days=days)
    df_tickers_processed = __full_handle_tickers(df_tickers, contract=contract)
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
