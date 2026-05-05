import argparse
import datetime
import io
import os
import time
import urllib.error
import urllib.request
import zipfile


DEFAULT_CONTEXT_TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "NEARUSDT")
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"


def parse_tickers(value):
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += datetime.timedelta(days=1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Directly download Binance USD-M daily metrics CSV files.")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--tickers", default=",".join(DEFAULT_CONTEXT_TICKERS))
    parser.add_argument("--start-date", default="2023-04-01")
    parser.add_argument("--end-date", default="2026-04-30")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args(argv)


def download_one(ticker, day, output_dir, retries, sleep_seconds):
    filename = f"{ticker}-metrics-{day.isoformat()}.csv"
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        return "cached"

    url = f"{BASE_URL}/{ticker}/{ticker}-metrics-{day.isoformat()}.zip"
    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
                if not csv_names:
                    return "empty_zip"
                with archive.open(csv_names[0]) as source, open(output_path, "wb") as target:
                    target.write(source.read())
            return "downloaded"
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return "missing"
            last_error = error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
        time.sleep(sleep_seconds * (attempt + 1))
    return f"failed:{type(last_error).__name__}"


def main(argv=None):
    args = parse_args(argv)
    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date)
    tickers = parse_tickers(args.tickers)

    for ticker in tickers:
        output_dir = os.path.join(args.data_dir, "futures", "um", "daily", "metrics", ticker)
        os.makedirs(output_dir, exist_ok=True)
        counts = {}
        for day in date_range(start, end):
            status = download_one(ticker, day, output_dir, args.retries, args.sleep)
            counts[status] = counts.get(status, 0) + 1
        print(f"{ticker}: {counts}")


if __name__ == "__main__":
    main()
