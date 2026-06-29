import tempfile
from pathlib import Path

import torch

from timeseries_forge.deployment.export import ExportableForgeNet, export_torchscript, load_torchscript


def test_exportable_forgenet_returns_tuple(small_model, small_config):
    wrapper = ExportableForgeNet(small_model)
    x = torch.randn(2, small_config.seq_len, small_config.num_features)
    forecast, reconstruction = wrapper(x)

    assert forecast.shape == (2, small_config.horizon, small_config.num_targets, len(small_config.quantiles))
    assert reconstruction.shape == (2, small_config.seq_len, small_config.num_features)


def test_torchscript_export_and_reload_matches_eager(small_model, small_config):
    small_model.eval()
    x = torch.randn(1, small_config.seq_len, small_config.num_features)

    with torch.no_grad():
        eager_forecast, eager_recon = ExportableForgeNet(small_model)(x)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_torchscript(small_model, x, Path(tmp) / "model.pt")
        assert path.exists()

        loaded = load_torchscript(path)
        with torch.no_grad():
            ts_forecast, ts_recon = loaded(x)

        assert torch.allclose(eager_forecast, ts_forecast, atol=1e-5)
        assert torch.allclose(eager_recon, ts_recon, atol=1e-5)


def test_torchscript_export_handles_different_batch_size(small_model, small_config):
    small_model.eval()
    example_input = torch.randn(1, small_config.seq_len, small_config.num_features)

    with tempfile.TemporaryDirectory() as tmp:
        path = export_torchscript(small_model, example_input, Path(tmp) / "model.pt")
        loaded = load_torchscript(path)

        # traced with batch size 1; tracing should still generalize across batch
        # dimension since no batch-size-dependent control flow exists in the model
        bigger_batch = torch.randn(4, small_config.seq_len, small_config.num_features)
        with torch.no_grad():
            forecast, recon = loaded(bigger_batch)
        assert forecast.shape[0] == 4
        assert recon.shape[0] == 4
