from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm

SHELLY_BASE_URL = "http://192.168.178.99"
PHASE_URL = SHELLY_BASE_URL + "/emeter/{}/em_data.csv"  # Replace {} with 0, 1, 2

CSV_PATH = Path(__file__).parent / "data/input"

logger = logging.getLogger(__name__)


def download_data():
    phases = [
        pd.read_csv(PHASE_URL.format(i), parse_dates=["Date/time UTC"])
        for i in tqdm(range(3))
    ]

    # Merge the dataframes
    df = pd.merge(phases[0], phases[1], on="Date/time UTC")
    df = pd.merge(df, phases[2], on="Date/time UTC")

    # Split to one dataframe per day
    df["Date"] = df["Date/time UTC"].dt.date
    for date in tqdm(df["Date"].unique(), desc="Saving data"):
        day_df = df[df["Date"] == date]
        if date == pd.Timestamp.now().date():
            file_path = CSV_PATH / f"current_day.csv"
        else:
            year, month = date.year, date.month
            file_path = CSV_PATH / f"{year}/{month:02}/{date}.csv"
            if file_path.exists():
                tqdm.write(f"Skipping {date} as it already exists")
                continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        day_df.to_csv(file_path, index=False, columns=[
            "Date/time UTC",
            "Active energy Wh (A)", "Returned energy Wh (A)",
            "Active energy Wh (B)", "Returned energy Wh (B)",
            "Active energy Wh (C)", "Returned energy Wh (C)",
        ])


def main(
        download=False,
        sample_rate: str | None = None,
        plot_phases=False,
        timeframe_start: str | None = None,
        timeframe_end: str | None = None,
        y_axis_limit: int | None = None,
):
    if download:
        download_data()

    if timeframe_start:
        timeframe_start = pd.Timestamp(timeframe_start)
    if timeframe_end:
        timeframe_end = pd.Timestamp(timeframe_end)

    if timeframe_start and timeframe_end and timeframe_start > timeframe_end:
        raise ValueError("timeframe_start must be before timeframe_end")

    # Read the data
    files = list(CSV_PATH.rglob("*.csv"))
    dfs = []
    for file in sorted(files):
        if (timeframe_start or timeframe_end) and file.stem != "current_day":
            file_date = pd.Timestamp(file.stem).date()
            if timeframe_start and timeframe_start.date() > file_date:
                continue
            if timeframe_end and timeframe_end.date() < file_date:
                continue
        print(f"Reading {file}")
        df = pd.read_csv(file, parse_dates=["Date/time UTC"])
        dfs.append(df)
    df = pd.concat(dfs)

    # Throw away data outside the timeframe
    if timeframe_start:
        df = df[df["Date/time UTC"] >= timeframe_start]
    if timeframe_end:
        df = df[df["Date/time UTC"] <= timeframe_end]

    # New column from sum of "Active energy Wh (A)", "Active energy Wh (B)", "Active energy Wh (C)"
    # and "Returned energy Wh (A)", "Returned energy Wh (B)", "Returned energy Wh (C)"
    df["Total Active energy Wh"] = df["Active energy Wh (A)"] + df["Active energy Wh (B)"] + df["Active energy Wh (C)"]
    df["Total Returned energy Wh"] = df["Returned energy Wh (A)"] + df["Returned energy Wh (B)"] + df[
        "Returned energy Wh (C)"]

    # Put the data into buckets. Sum the values in each bucket.
    ts = df.set_index("Date/time UTC")
    if sample_rate:
        ts = ts.resample(sample_rate).sum()

    # Plot the data
    if plot_phases:
        ax = ts.plot.area(
            y=["Active energy Wh (A)", "Active energy Wh (B)", "Active energy Wh (C)"],
            title="Active energy Wh per phase",
            xlabel="Date/time UTC",
            ylabel="Energy Wh",
            figsize=(20, 10),
            grid=True,
            ylim=(0, y_axis_limit) if y_axis_limit else None,
        )
        ts.plot.area(
            ax=ax,
            y=["Returned energy Wh (A)", "Returned energy Wh (B)", "Returned energy Wh (C)"],
            title="Returned energy Wh per phase",
            xlabel="Date/time UTC",
            ylabel="Energy Wh",
            figsize=(20, 10),
            grid=True,
            ylim=(0, y_axis_limit) if y_axis_limit else None,
        )
    else:
        ts.plot(y=["Total Active energy Wh", "Total Returned energy Wh"],
                title="Total Active and Returned energy Wh",
                xlabel="Date/time UTC",
                ylabel="Energy Wh",
                figsize=(20, 10),
                grid=True,
                style=["-", ":", "--", "-.", "-", ":", "--", "-."],
                drawstyle="steps-post",
                ylim=(0, y_axis_limit) if y_axis_limit else None,
                )

    # Save the plot
    plt.savefig("data/plot.png")

    print(f"Total Active energy Wh: {ts['Total Active energy Wh'].sum()}")
    print(f"Total Returned energy Wh: {ts['Total Returned energy Wh'].sum()}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--download", action="store_true", help="Download the data from the Shelly device")
    args.add_argument("--sample-rate", type=str, help="Sample rate for the data")
    args.add_argument("--plot-phases", action="store_true", help="Plot the data for each phase")
    args.add_argument("--timeframe-start", type=str, help="Plot the graph starting at this datetime")
    args.add_argument("--timeframe-end", type=str, help="Plot the graph ending at this datetime")
    args.add_argument("--y-axis-limit", type=int, help="Limit the y-axis to this value")
    args = args.parse_args()
    main(args.download, args.sample_rate, args.plot_phases, args.timeframe_start, args.timeframe_end, args.y_axis_limit)
