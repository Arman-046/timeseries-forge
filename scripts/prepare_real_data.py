#!/usr/bin/env python3
"""
Download and prepare the Beijing PM2.5 dataset (UCI ML Repository,
dataset id 381) into the (time, channels) numpy format expected by
SlidingWindowDataset.

This is real, public, hourly multivariate sensor data -- pollution
concentration plus meteorological covariates -- collected at the US
Embassy in Beijing, Jan 2010 - Dec 2014. It's a genuine stand-in for
the kind of IoT/industrial sensor telemetry this project targets, and
gives the README a real (not just synthetic) forecasting result.

Source / citation:
    Liang, X. et al. (2015). Assessing Beijing's PM2.5 pollution:
    severity, weather impact, APEC and winter heating. Proceedings of
    the Royal Society A. UCI ML Repository, dataset id 381.
    https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
    Licensed for research/educational use via UCI ML Repository.

Usage:
    pip install ucimlrepo
    python scripts/prepare_real_data.py --output-path data/beijing_pm25.npy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("prepare_real_data")

# Numeric columns used as model channels. `cbwd` (combined wind direction)
# is categorical text (e.g. "NE", "SE") and is one-hot encoded separately
# rather than included here directly.
NUMERIC_COLUMNS = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
WIND_DIRECTION_COLUMN = "cbwd"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare the Beijing PM2.5 dataset for ForgeNet")
    p.add_argument("--output-path", type=str, default="data/beijing_pm25.npy")
    p.add_argument(
        "--include-wind-direction",
        action="store_true",
        help="one-hot encode the categorical wind-direction column as extra channels",
    )
    p.add_argument(
        "--fill-method",
        choices=["interpolate", "ffill", "drop"],
        default="interpolate",
        help="how to handle missing values (the raw data has NA gaps, mostly in pm2.5)",
    )
    return p.parse_args()


def load_raw_dataframe() -> pd.DataFrame:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError(
            "this script requires the `ucimlrepo` package: pip install ucimlrepo"
        ) from e

    logger.info("fetching Beijing PM2.5 dataset (UCI id=381)...")
    dataset = fetch_ucirepo(id=381)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    return df


def prepare_array(
    df: pd.DataFrame, include_wind_direction: bool, fill_method: str
) -> tuple[np.ndarray, list[str]]:
    # ensure chronological order -- the raw file should already be sorted,
    # but this guards against fetch_ucirepo ever changing row order
    sort_cols = [c for c in ["year", "month", "day", "hour"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    missing_before = df[NUMERIC_COLUMNS].isna().sum().sum()
    logger.info("missing values across numeric columns before fill: %d", missing_before)

    if fill_method == "interpolate":
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].interpolate(limit_direction="both")
    elif fill_method == "ffill":
        df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].ffill().bfill()
    elif fill_method == "drop":
        df = df.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)

    channel_names = list(NUMERIC_COLUMNS)
    arrays = [df[NUMERIC_COLUMNS].to_numpy(dtype=np.float32)]

    if include_wind_direction and WIND_DIRECTION_COLUMN in df.columns:
        one_hot = pd.get_dummies(df[WIND_DIRECTION_COLUMN], prefix="wind")
        arrays.append(one_hot.to_numpy(dtype=np.float32))
        channel_names.extend(one_hot.columns.tolist())

    data = np.concatenate(arrays, axis=1)

    remaining_nan = np.isnan(data).sum()
    if remaining_nan > 0:
        logger.warning(
            "%d NaNs remain after fill (likely a leading/trailing gap); "
            "trimming those rows", remaining_nan
        )
        valid_rows = ~np.isnan(data).any(axis=1)
        data = data[valid_rows]

    return data, channel_names


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    df = load_raw_dataframe()
    data, channel_names = prepare_array(df, args.include_wind_direction, args.fill_method)

    logger.info("final array shape: %s, channels: %s", data.shape, channel_names)
    logger.info(
        "suggested training command:\n"
        "  python scripts/train.py --data-path %s --target-indices 0 2 "
        "--seq-len 168 --horizon 24",
        args.output_path,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data)

    channels_path = out_path.with_suffix(".channels.txt")
    channels_path.write_text("\n".join(channel_names))
    logger.info("saved data to %s and channel names to %s", out_path, channels_path)


if __name__ == "__main__":
    main()
