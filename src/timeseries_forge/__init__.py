"""
TimeSeries Forge
================

A production-oriented PyTorch library for multivariate time series
forecasting and anomaly detection using a shared Temporal Fusion
Transformer-style encoder with a probabilistic forecasting head and
a reconstruction-based anomaly detection head.

See README.md for full documentation.
"""

__version__ = "0.1.0"

__all__ = ["ForgeNet", "ForgeNetConfig", "__version__"]


def __getattr__(name: str):
    # Lazily import ForgeNet/ForgeNetConfig (PEP 562) so that torch-free
    # submodules -- e.g. data.synthetic, evaluation.anomaly_metrics, or
    # scripts.prepare_real_data -- can be imported and used without
    # requiring torch to be installed at all. `import timeseries_forge`
    # alone, or `from timeseries_forge import ForgeNet`, still works
    # exactly as before; only the *unconditional* eager import is removed.
    if name in ("ForgeNet", "ForgeNetConfig"):
        from timeseries_forge.models.forge_net import ForgeNet, ForgeNetConfig

        return {"ForgeNet": ForgeNet, "ForgeNetConfig": ForgeNetConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
