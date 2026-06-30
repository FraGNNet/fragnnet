import sys
from pathlib import Path

# Ensure repository root is on sys.path so `scripts` package can be imported.
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import scripts.wandb_scripts.wandb_cleanup as wandb_cleanup


class DummyFile:
    def __init__(self, name: str, size: int = 1024, updated_at: str = "2020-01-01T00:00:00Z"):
        self.name = name
        self.size = size
        self.updated_at = updated_at
        self.deleted = False

    def delete(self):
        self.deleted = True


class DummyRun:
    def __init__(
        self, name: str, run_id: str, files: list[DummyFile], tags: list[str] | None = None
    ):
        self.name = name
        self.id = run_id
        self.tags = tags
        self._files = files

    def files(self, per_page: int = 100):
        return iter(self._files)


class DummyApi:
    def __init__(self, runs: list[DummyRun]):
        self._runs = runs

    def runs(self, path: str):
        return self._runs


def test_main_deletes_last_ckpt_when_it_is_the_only_checkpoint(monkeypatch, capsys):
    last_ckpt = DummyFile("last.ckpt")
    run = DummyRun("example-run", "run-1", [last_ckpt])
    api = DummyApi([run])

    monkeypatch.setattr(wandb_cleanup.wandb, "Api", lambda: api, raising=False)
    monkeypatch.setattr(sys, "argv", ["wandb_cleanup.py"])

    wandb_cleanup.main()

    output = capsys.readouterr().out
    assert "No epoch checkpoints found; deleting all .ckpt files" in output
    assert "last.ckpt" in output
    assert "WOULD DELETE" in output


def test_main_keeps_epoch_checkpoint_and_deletes_last_ckpt(monkeypatch, capsys):
    epoch_ckpt = DummyFile("model-epoch=3.ckpt")
    last_ckpt = DummyFile("last.ckpt")
    run = DummyRun("example-run", "run-2", [epoch_ckpt, last_ckpt])
    api = DummyApi([run])

    monkeypatch.setattr(wandb_cleanup.wandb, "Api", lambda: api, raising=False)
    monkeypatch.setattr(sys, "argv", ["wandb_cleanup.py"])

    wandb_cleanup.main()

    output = capsys.readouterr().out
    assert "Kept epoch 3" in output
    assert "last.ckpt" in output
    assert "WOULD DELETE" in output
