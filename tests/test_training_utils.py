import tempfile
from pathlib import Path

import torch

from timeseries_forge.training.checkpoint import CheckpointManager, EarlyStopping
from timeseries_forge.training.scheduler import cosine_warmup_scheduler


def test_cosine_warmup_scheduler_ramps_up_then_down():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_warmup_scheduler(opt, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    # LR should increase during warmup
    assert lrs[9] > lrs[0]
    # LR should be lower at the end (cosine decay) than at the peak
    assert lrs[-1] < max(lrs)


def test_cosine_warmup_scheduler_respects_min_lr_ratio():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0)
    sched = cosine_warmup_scheduler(opt, warmup_steps=5, total_steps=50, min_lr_ratio=0.1)

    for _ in range(60):  # go past total_steps
        opt.step()
        sched.step()

    final_lr = opt.param_groups[0]["lr"]
    assert final_lr >= 0.1 - 1e-6


def test_early_stopping_tracks_best_and_triggers():
    es = EarlyStopping(patience=3, mode="min")
    values = [1.0, 0.9, 0.95, 0.96, 0.97]  # improves once, then stalls
    results = [es.step(v) for v in values]

    assert results[0] is True  # first value always "improves"
    assert results[1] is True  # 0.9 < 1.0
    assert results[2] is False  # 0.95 not better than 0.9
    assert es.should_stop is True  # patience=3 exceeded by the time we reach 0.97


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert es.step(0.5) is True
    assert es.step(0.6) is True  # improvement
    assert es.step(0.55) is False  # no improvement
    assert es.step(0.55) is False
    assert es.should_stop is True


def test_checkpoint_manager_save_and_load_round_trip():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_warmup_scheduler(opt, warmup_steps=2, total_steps=10)

    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp)
        mgr.save(model, opt, sched, epoch=1, metrics={"val_loss": 0.5}, is_best=True)

        assert (Path(tmp) / "best.pt").exists()
        assert (Path(tmp) / "last.pt").exists()
        assert (Path(tmp) / "metadata.json").exists()

        loaded = mgr.load()
        assert loaded["epoch"] == 1
        assert loaded["metrics"]["val_loss"] == 0.5
        # state dict should be loadable back into a fresh model
        fresh_model = torch.nn.Linear(4, 4)
        fresh_model.load_state_dict(loaded["model_state_dict"])


def test_checkpoint_manager_only_overwrites_best_when_flagged():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp)
        mgr.save(model, opt, None, epoch=1, metrics={"val_loss": 0.5}, is_best=True)
        best_v1 = mgr.load(mgr.best_path)

        mgr.save(model, opt, None, epoch=2, metrics={"val_loss": 0.9}, is_best=False)
        best_v2 = mgr.load(mgr.best_path)

        # best checkpoint should still reflect epoch 1 since epoch 2 wasn't best
        assert best_v1["epoch"] == best_v2["epoch"] == 1
