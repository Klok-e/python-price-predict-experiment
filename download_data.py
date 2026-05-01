import argparse
import datetime

from utils.util import BINANCE_DATA_START_DATE, DEFAULT_TICKERS, download_ohlc_data


def parse_tickers(value):
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download Binance 1m OHLC data into the local cache.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--start-date", default=BINANCE_DATA_START_DATE.isoformat())
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tickers = parse_tickers(args.tickers)
    start_date = datetime.date.fromisoformat(args.start_date)
    download_ohlc_data(args.data_dir, tickers=tickers, start_date=start_date)
    print(f"downloaded/updated {len(tickers)} tickers in {args.data_dir} from {start_date}")


if __name__ == "__main__":
    main()
