import numpy as np
import torch

from timeseries_forge.data.datasets import (
    ChannelScaler,
    SlidingWindowDataset,
    train_val_test_split_indices,
)
from timeseries_forge.data.synthetic import generate_synthetic_series


def test_channel_scaler_round_trip():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=3.0, size=(100, 4))
    scaler = ChannelScaler().fit(x)
    scaled = scaler.transform(x)

    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-6)

    restored = scaler.inverse_transform(scaled)
    assert np.allclose(restored, x, atol=1e-4)


def test_channel_scaler_state_dict_round_trip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    scaler = ChannelScaler().fit(x)
    state = scaler.state_dict()
    restored = ChannelScaler.from_state_dict(state)

    assert np.allclose(restored.mean, scaler.mean)
    assert np.allclose(restored.std, scaler.std)


def test_train_val_test_split_is_chronological_and_covers_all_data():
    n = 1000
    train_s, val_s, test_s = train_val_test_split_indices(n, train_frac=0.7, val_frac=0.15)

    assert train_s.start == 0
    assert train_s.stop == val_s.start
    assert val_s.stop == test_s.start
    assert test_s.stop == n


def test_sliding_window_dataset_shapes():
    data = np.random.randn(200, 4).astype(np.float32)
    ds = SlidingWindowDataset(data, seq_len=20, horizon=5, target_indices=[0, 1])

    expected_n_windows = 200 - 20 - 5 + 1
    assert len(ds) == expected_n_windows

    sample = ds[0]
    assert sample["input"].shape == (20, 4)
    assert sample["forecast_target"].shape == (5, 2)
    assert sample["reconstruction_target"].shape == (20, 4)


def test_sliding_window_dataset_raises_on_too_short_series():
    data = np.random.randn(10, 4).astype(np.float32)
    try:
        SlidingWindowDataset(data, seq_len=20, horizon=5, target_indices=[0])
        assert False, "expected ValueError for series shorter than seq_len + horizon"
    except ValueError:
        pass


def test_sliding_window_dataset_noise_changes_input_but_not_target():
    data = np.ones((50, 2), dtype=np.float32) * 3.0
    ds = SlidingWindowDataset(data, seq_len=10, horizon=3, target_indices=[0], noise_std=1.0)
    torch.manual_seed(0)
    sample = ds[0]
    # input should be perturbed away from the constant value (extremely unlikely to match exactly)
    assert not torch.allclose(sample["input"], torch.full((10, 2), 3.0))
    # reconstruction target should remain the clean window
    assert torch.allclose(sample["reconstruction_target"], torch.full((10, 2), 3.0))


def test_synthetic_series_shapes_and_label_range():
    data, labels = generate_synthetic_series(n_steps=500, n_channels=3, anomaly_rate=0.02)
    assert data.shape == (500, 3)
    assert labels.shape == (500,)
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels.sum() > 0  # at least some anomalies were injected
