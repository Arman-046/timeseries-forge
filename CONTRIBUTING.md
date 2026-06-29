# Contributing

Thanks for considering a contribution! This is a personal portfolio
project, but issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Arman-046/timeseries-forge.git
cd timeseries-forge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
```

## Before opening a PR

```bash
black src tests scripts
ruff check src tests scripts --fix
pytest --cov=timeseries_forge
```

All three should pass cleanly — this matches exactly what CI
(`.github/workflows/ci.yml`) runs on every push and pull request.

## Code style

- Type hints on all public function signatures.
- Docstrings should explain *why*, not just *what* — restating the
  code in prose adds nothing; explaining the design tradeoff does.
- New model components go in `src/timeseries_forge/models/`; add a
  corresponding test in `tests/test_layers.py` or `tests/test_forge_net.py`
  covering at minimum output shape and gradient flow.
- New metrics go in `src/timeseries_forge/evaluation/` and should be
  tested against at least one hand-constructed edge case (perfect
  prediction, all-zero prediction, etc.), not only random data.
