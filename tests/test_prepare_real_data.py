import numpy as np
import pandas as pd

from scripts.prepare_real_data import NUMERIC_COLUMNS, WIND_DIRECTION_COLUMN, prepare_array


def _make_fake_dataframe(n=50, with_nans=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "year": 2013,
            "month": 1,
            "day": (np.arange(n) // 24) + 1,
            "hour": np.arange(n) % 24,
            "pm2.5": rng.uniform(0, 300, n),
            "DEWP": rng.uniform(-20, 20, n),
            "TEMP": rng.uniform(-10, 35, n),
            "PRES": rng.uniform(990, 1040, n),
            "Iws": rng.uniform(0, 50, n),
            "Is": rng.integers(0, 5, n).astype(float),
            "Ir": rng.integers(0, 5, n).astype(float),
            "cbwd": rng.choice(["NE", "SE", "NW", "cv"], size=n),
        }
    )
    if with_nans:
        df.loc[5:7, "pm2.5"] = np.nan
        df.loc[20, "TEMP"] = np.nan
    return df


def test_prepare_array_basic_shape_no_wind_direction():
    df = _make_fake_dataframe(n=50, with_nans=False)
    data, channel_names = prepare_array(df, include_wind_direction=False, fill_method="interpolate")

    assert data.shape == (50, len(NUMERIC_COLUMNS))
    assert channel_names == NUMERIC_COLUMNS
    assert not np.isnan(data).any()


def test_prepare_array_interpolates_missing_values():
    df = _make_fake_dataframe(n=50, with_nans=True)
    data, _ = prepare_array(df, include_wind_direction=False, fill_method="interpolate")
    assert not np.isnan(data).any()
    assert data.shape[0] == 50  # interpolation should not drop rows


def test_prepare_array_drop_fill_method_removes_nan_rows():
    df = _make_fake_dataframe(n=50, with_nans=True)
    data, _ = prepare_array(df, include_wind_direction=False, fill_method="drop")
    assert not np.isnan(data).any()
    assert data.shape[0] < 50  # rows with NaNs should be dropped


def test_prepare_array_with_wind_direction_one_hot():
    df = _make_fake_dataframe(n=50, with_nans=False)
    data, channel_names = prepare_array(df, include_wind_direction=True, fill_method="interpolate")

    n_wind_categories = df[WIND_DIRECTION_COLUMN].nunique()
    assert data.shape[1] == len(NUMERIC_COLUMNS) + n_wind_categories
    assert len(channel_names) == data.shape[1]
    # one-hot columns should be binary (0/1)
    one_hot_part = data[:, len(NUMERIC_COLUMNS):]
    assert set(np.unique(one_hot_part)).issubset({0.0, 1.0})
    # each row's one-hot block should sum to exactly 1 (exactly one wind direction)
    assert np.allclose(one_hot_part.sum(axis=1), 1.0)


def test_prepare_array_sorts_chronologically_even_if_input_shuffled():
    df = _make_fake_dataframe(n=50, with_nans=False)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    data_sorted, _ = prepare_array(df, include_wind_direction=False, fill_method="interpolate")
    data_from_shuffled, _ = prepare_array(shuffled, include_wind_direction=False, fill_method="interpolate")

    assert np.allclose(data_sorted, data_from_shuffled)
