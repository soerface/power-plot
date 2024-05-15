import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging

from pandas import DataFrame
from tqdm import tqdm
import paramiko

logger = logging.getLogger(__name__)


def write_to_fs(csv_path: Path, df: DataFrame):
    (csv_path / "current_day.csv").unlink(missing_ok=True)
    # Split to one dataframe per day
    for date in tqdm(df["Date"].unique(), desc="Saving data"):
        logger.debug(f"Saving data for {date}")
        day_df = df[df["Date"] == date]
        if date == pd.Timestamp.now(tz="UTC").date():
            file_path = csv_path / f"current_day.csv"
        else:
            year, month = date.year, date.month
            file_path = csv_path / f"{year}/{month:02}/{date}.csv"
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


def write_to_sftp(csv_path: str, df: DataFrame, ssh_key_path: str):
    assert csv_path.startswith("sftp://")
    sftp_url = csv_path[len("sftp://"):]
    hostname, _, path = sftp_url.partition("/")
    hostname, _, ssh_port = hostname.partition(":")
    username, _, hostname = hostname.rpartition("@")
    if not username:
        logger.error("Username is required in the SFTP URL (sftp://username@hostname:port/path)")
        sys.exit(1)

    transport = paramiko.Transport((hostname, int(ssh_port) or 22))
    try:
        transport.connect(username=username, pkey=paramiko.RSAKey.from_private_key_file(ssh_key_path))
    except paramiko.ssh_exception.AuthenticationException:
        logger.error("Authentication failed. Check your credentials")
        sys.exit(1)
    sftp = paramiko.SFTPClient.from_transport(transport)
    pbar = tqdm(df["Date"].unique())
    for date in pbar:
        pbar.set_description(f"Saving data for {date}")
        try:
            sftp.chdir(f"/{path}")
        except FileNotFoundError:
            logger.error(f"Path /{path} does not exist on the SFTP server")
            logger.info(f"Existing paths: {sftp.listdir()}")
            sys.exit(1)
        day_df = df[df["Date"] == date]
        if date == pd.Timestamp.now(tz="UTC").date():
            file_path = "current_day.csv"
        else:
            year, month = date.year, date.month
            file_path = f"{year}/{month:02}/{date}.csv"
            try:
                sftp.stat(file_path)
                tqdm.write(f"Skipping {date} as the file sftp://{hostname}/{path}/{file_path} already exists")
                continue
            except FileNotFoundError:
                pass
        path_components, _, filename = file_path.rpartition("/")
        for path_component in path_components.split("/"):
            # TODO: optimize speed by not constantly changing directories
            try:
                sftp.chdir(path_component)
            except IOError:
                sftp.mkdir(path_component)
                sftp.chdir(path_component)
        with sftp.file(filename, "w") as file:
            tqdm.write(f"Writing data to /{path}/{file_path}")
            day_df.to_csv(file, index=False, columns=[
                "Date/time UTC",
                "Active energy Wh (A)", "Returned energy Wh (A)",
                "Active energy Wh (B)", "Returned energy Wh (B)",
                "Active energy Wh (C)", "Returned energy Wh (C)",
            ])
    sftp.close()
    transport.close()


def download_data(hostname: str, csv_path: str, ssh_key_path: str | None = None):
    if csv_path.startswith("sftp://") and not ssh_key_path:
        logger.error("--ssh-key-path must be provided when using SFTP")
        sys.exit(1)
    if ssh_key_path and not Path(ssh_key_path).exists():
        logger.error(f"SSH key at {ssh_key_path} does not exist")
        sys.exit(1)

    if not hostname:
        logger.error("--host is required when downloading data")
        sys.exit(1)

    phase_url = f"http://{hostname}/emeter/%d/em_data.csv"
    phases = [
        pd.read_csv(phase_url % i, parse_dates=["Date/time UTC"])
        for i in tqdm(range(3), desc=f"Downloading CSV from {hostname}")
    ]

    # Merge the dataframes
    df = pd.merge(phases[0], phases[1], on="Date/time UTC", how="outer")
    df = pd.merge(df, phases[2], on="Date/time UTC", how="outer")

    # Sort the data by date
    df = df.sort_values("Date/time UTC")

    df["Date"] = df["Date/time UTC"].dt.date
    if csv_path.startswith("sftp://"):
        write_to_sftp(csv_path, df, ssh_key_path)
    else:
        write_to_fs(Path(csv_path), df)


def read_from_fs(csv_path: Path, timeframe_start: str | None, timeframe_end: str | None) -> DataFrame:
    files = list(csv_path.rglob("*.csv"))
    dfs = []
    for file in sorted(files):
        if (timeframe_start or timeframe_end) and file.stem != "current_day":
            file_date = pd.Timestamp(file.stem).date()
            if timeframe_start and timeframe_start.date() > file_date:
                continue
            if timeframe_end and timeframe_end.date() < file_date:
                continue
        df = pd.read_csv(file, parse_dates=["Date/time UTC"])
        dfs.append(df)
    return pd.concat(dfs)


def main(
        csv_path: str,
        output_path: Path,
        sample_rate: str = "1min",
        plot_phases=False,
        timeframe_start: str | None = None,
        timeframe_end: str | None = None,
        y_axis_limit: int | None = None,
        dark_theme=False,
):
    if timeframe_start:
        timeframe_start = pd.Timestamp(timeframe_start)
    if timeframe_end:
        timeframe_end = pd.Timestamp(timeframe_end)

    if timeframe_start and timeframe_end and timeframe_start > timeframe_end:
        raise ValueError("timeframe_start must be before timeframe_end")

    # Read the data
    if csv_path.startswith("sftp://"):
        raise NotImplementedError("Reading from SFTP not supported yet")
    else:
        df = read_from_fs(Path(csv_path), timeframe_start, timeframe_end)

    # TODO: Convert UTC to local timezone
    # df["Date/time UTC"] = df["Date/time UTC"].dt.tz_convert("Europe/Berlin")

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

    ts = df.set_index("Date/time UTC")
    # Put the data into buckets. Sum the values in each bucket.
    ts = ts.resample(sample_rate).sum()

    # Plot the data
    if dark_theme:
        plt.style.use("dark_background")
        # colors of the graph should look normal
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf"
        ])
    ax = ts.plot(
        y=["Total Active energy Wh", "Total Returned energy Wh"],
        title="Total Active and Returned energy Wh",
        xlabel="Date/time UTC",
        ylabel="Energy Wh",
        figsize=(20, 10),
        grid=True,
        style=["-", ":", "--", "-.", "-", ":", "--", "-."],
        drawstyle="steps-post",
        # TODO: ylim start from 0 probably doesn't work with returned energy
        ylim=(0, y_axis_limit) if y_axis_limit else None,
    )
    if plot_phases:
        columns = ["Active energy Wh (A)", "Active energy Wh (B)", "Active energy Wh (C)",
                      "Returned energy Wh (A)", "Returned energy Wh (B)", "Returned energy Wh (C)"]
        for i, (column, color) in enumerate(zip(
                columns,
                ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd", "#8c564b"],
        )):
            y0 = ts[columns[:i]].sum(axis=1)
            ax.fill_between(
                ts.index,
                y0,
                y0 + ts[column],
                label=column,
                color=color,
                step="post",
            )
        ax.legend()
    else:
        ax.fill_between(
            ts.index,
            ts["Total Active energy Wh"],
            ts["Total Returned energy Wh"],
            label="Active energy Wh",
            color="#1f77b4",
            step="post",
        )

    # Save the plot
    plt.savefig(output_path)

    print(f"Total Active energy Wh: {ts['Total Active energy Wh'].sum()}")
    print(f"Total Returned energy Wh: {ts['Total Returned energy Wh'].sum()}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args.add_argument(
        "csv_path",
        type=str,
        help="Directory to read the CSV files from. "
             "Also supports sftp:// URLs for writing to an SFTP server. "
             "If combined with --download, the downloaded files will be saved here.",
    )
    args.add_argument(
        "--output_path", "-o",
        type=Path,
        help="Path to save the plot to. "
             "File ending determines the format. "
             "Format must be supported by matplotlib."
             ".png and .svg at least are supported.",
    )
    args.add_argument("--download", action="store_true", help="Download the data from the Shelly device")
    args.add_argument(
        "--download-only",
        action="store_true",
        help="Do not plot, only download"
    )
    args.add_argument("--ssh-key-path", type=str, help="Path to the SSH key for SFTP")
    args.add_argument("--ssh-port", type=int, help="Port for SFTP", default=22)
    args.add_argument("--host", type=str, help="Hostname or IP address of the Shelly device")
    args.add_argument("--sample-rate", type=str, help="Sample rate for the data", default="1min")
    args.add_argument("--plot-phases", action="store_true", help="Plot the data for each phase")
    args.add_argument("--from", type=str, help="Plot the graph starting at this datetime")
    args.add_argument("--to", type=str, help="Plot the graph ending at this datetime")
    args.add_argument("--y-lim", type=int, help="Limit the y-axis to this value")
    args.add_argument("--dark-theme", action="store_true", help="Use a dark theme for the plot")
    args.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level. Repeat for more verbosity")
    args = args.parse_args()

    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter("%(asctime)s %(name)s - %(levelname)s: %(message)s")
    log_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    root_logger.setLevel(logging.ERROR - args.verbose * 10)
    logger.info("Loglevel set to %s", logging.getLevelName(logger.getEffectiveLevel()))

    if args.download or args.download_only:
        download_data(args.host, args.csv_path, args.ssh_key_path)
    if not args.download_only:
        if not args.output_path:
            logger.error("--output_path is required when plotting")
            sys.exit(1)
        main(
            args.csv_path,
            args.output_path,
            args.sample_rate,
            args.plot_phases,
            getattr(args, "from"),
            args.to,
            args.y_lim,
            args.dark_theme,
        )
