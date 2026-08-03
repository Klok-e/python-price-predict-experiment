import argparse
import datetime

from binance_historical_data import BinanceDataDumper


DEFAULT_CONTEXT_TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def parse_tickers(value):
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download Binance USD-M futures metrics into the local cache.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--tickers", default=",".join(DEFAULT_CONTEXT_TICKERS))
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    date_start = datetime.date.fromisoformat(args.start_date)
    date_end = datetime.date.fromisoformat(args.end_date) if args.end_date else None
    tickers = parse_tickers(args.tickers)

    dumper = BinanceDataDumper(
        path_dir_where_to_dump=f"{args.data_dir}/",
        asset_class="um",
        data_type="metrics",
    )
    dumper.dump_data(
        tickers=tickers,
        date_start=date_start,
        date_end=date_end,
        is_to_update_existing=True,
    )
    print(f"downloaded/updated USD-M futures metrics for {len(tickers)} tickers in {args.data_dir}")


if __name__ == "__main__":
    main()
