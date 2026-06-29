"""
Model export utilities: TorchScript (via tracing) and ONNX.

ForgeNet.forward returns a dict and accepts an optional argument,
neither of which export cleanly to TorchScript/ONNX in general. This
module wraps the trained model in a fixed-signature, tuple-output
module (`ExportableForgeNet`) that is trace-friendly, then exports
that wrapper. Tracing (rather than scripting) is used because the
model's control flow (e.g. the optional causal mask branch) is fully
determined by the config at construction time and does not depend on
tensor *values* at runtime, which is exactly the condition under which
tracing is safe and scripting's extra rigidity isn't needed.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from timeseries_forge.models.forge_net import ForgeNet


class ExportableForgeNet(nn.Module):
    """Wraps ForgeNet with a fixed (tensor in) -> (tensor, tensor) out signature.

    Returns (forecast, reconstruction) only -- attention/variable
    weights are dropped for the exported artifact since they're an
    interpretability aid for training-time analysis, not needed by a
    production inference client, and dropping them keeps the exported
    graph smaller and the output contract stable.
    """

    def __init__(self, model: ForgeNet):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        return outputs["forecast"], outputs["reconstruction"]


def export_torchscript(
    model: ForgeNet, example_input: torch.Tensor, out_path: str | Path
) -> Path:
    """Traces the model and saves a TorchScript artifact (.pt).

    The traced module has no Python dependency at load time -- it can
    be loaded from C++ (LibTorch) or Python without timeseries_forge
    installed, which is the whole point of TorchScript for deployment
    into environments that shouldn't carry full training code/deps.
    """
    wrapper = ExportableForgeNet(model)
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_input)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    return out_path


def export_onnx(
    model: ForgeNet,
    example_input: torch.Tensor,
    out_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Exports to ONNX with dynamic batch and sequence-length axes.

    Dynamic axes matter here specifically because a real deployment
    will see variable batch sizes (online single-sample inference vs.
    batched backfill jobs) and, if the serving window length ever
    changes, a variable sequence length too -- without dynamic_axes the
    exported graph would silently fail or require re-export for every
    shape combination.
    """
    wrapper = ExportableForgeNet(model)
    wrapper.eval()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        example_input,
        str(out_path),
        input_names=["input_window"],
        output_names=["forecast", "reconstruction"],
        dynamic_axes={
            "input_window": {0: "batch", 1: "seq_len"},
            "forecast": {0: "batch"},
            "reconstruction": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset_version,
    )
    return out_path


def load_torchscript(path: str | Path, device: str = "cpu") -> torch.jit.ScriptModule:
    return torch.jit.load(str(path), map_location=device)
