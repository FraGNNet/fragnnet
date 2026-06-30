from pathlib import Path

import pandas as pd

import scripts.wandb_scripts.wandb_exporter as we


class DummyRun:
    def __init__(self, name=None, path="user/proj/runid", df=None, rows=None, summary=None):
        self.name = name
        self.path = path
        self._df = df
        self._rows = rows or []
        self.summary = summary or {}

    def history(self, pandas: bool = False):
        if pandas:
            if self._df is None:
                raise RuntimeError("no pandas history")
            return self._df
        return iter(self._rows)


class DummyApi:
    def __init__(self, run_obj):
        self._run = run_obj

    def run(self, run_path: str):
        return self._run


def test_export_wandb_history_with_dataframe(monkeypatch, tmp_path):
    df = pd.DataFrame({"loss": [0.1, 0.05, 0.02], "note": ["a", "b", "c"]})
    run = DummyRun(name="testrun", df=df, summary={"best": 0.02})
    api = DummyApi(run)

    monkeypatch.setattr(we.wandb, "Api", lambda: api, raising=False)

    out = we.export_wandb_history(run_path="owner/proj/r", out_dir=str(tmp_path), spark_width=10)

    assert out.endswith(".txt")
    txt = Path(out).read_text(encoding="utf-8")
    assert "Run history:" in txt
    assert "loss" in txt
    assert "Run summary:" in txt
    assert "best" in txt


def test_export_wandb_history_fallback_list(monkeypatch, tmp_path):
    rows = [{"iter": 1, "metric": 0.5}, {"iter": 2, "metric": 0.3}]
    run = DummyRun(name=None, path="owner/proj/run42", df=None, rows=rows, summary={"m": 0.3})
    api = DummyApi(run)

    monkeypatch.setattr(we.wandb, "Api", lambda: api, raising=False)

    out = we.export_wandb_history(run_path="owner/proj/run42", out_dir=str(tmp_path), spark_width=5)

    assert out.endswith(".txt")
    txt = Path(out).read_text(encoding="utf-8")
    assert "iter" in txt or "metric" in txt
    assert "Run summary:" in txt
