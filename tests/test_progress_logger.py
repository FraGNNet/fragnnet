"""Tests for ProgressLogger, the shared interval-based progress reporter."""

import logging

from fragnnet.utils.misc_utils import ProgressLogger


def test_logs_on_item_interval(caplog):
    """A line is emitted every log_every items, not on every item."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=10, desc="Frag gen", log_every=5, log_seconds=1e9)
        for _ in range(10):
            progress.update()

    headers = [r.message for r in caplog.records if r.message.startswith("Frag gen:")]
    assert headers == ["Frag gen: 5/10 (50.0%)", "Frag gen: 10/10 (100.0%)"]


def test_logs_on_time_interval(caplog):
    """A line is emitted once log_seconds has elapsed even below log_every."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=100, desc="Frag gen", log_every=1000, log_seconds=0.0)
        progress.update()

    assert any(r.message == "Frag gen: 1/100 (1.0%)" for r in caplog.records)


def test_errors_counted_and_warned(caplog):
    """Errors are tallied and surfaced at WARNING level in the summary."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=4, desc="Frag gen", log_every=1000, log_seconds=1e9)
        for i in range(4):
            progress.update(is_error=(i < 2))
        progress.finish()

    assert progress.n_done == 4
    assert progress.n_errors == 2
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert "Total errors encountered: 2" in warnings
    assert "Success rate: 50.0%" in warnings


def test_clean_run_reports_full_success(caplog):
    """With no errors the summary says so instead of emitting a success rate."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=3, desc="Frag gen", log_every=1000, log_seconds=1e9)
        for _ in range(3):
            progress.update()
        progress.finish()

    messages = [r.message for r in caplog.records]
    assert "No errors encountered - 100% success rate" in messages
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_finish_on_empty_run_does_not_divide_by_zero(caplog):
    """total=0 with no updates must not raise (empty shard in an array job)."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=0, desc="Frag gen")
        progress.finish()

    assert "Total items processed: 0" in [r.message for r in caplog.records]


def test_update_accepts_batch_increments(caplog):
    """update(n=...) advances by n so callers can report in batches."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=10, desc="Frag gen", log_every=4, log_seconds=1e9)
        progress.update(n=4)
        progress.update(n=4)

    assert progress.n_done == 8
    headers = [r.message for r in caplog.records if r.message.startswith("Frag gen:")]
    assert headers == ["Frag gen: 4/10 (40.0%)", "Frag gen: 8/10 (80.0%)"]


def test_respects_injected_logger(caplog):
    """Records are emitted through the caller's logger, not misc_utils'."""
    logger = logging.getLogger("frag_gen_test_channel")
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(total=1, desc="Frag gen", log_every=1, logger=logger)
        progress.update()

    assert any(r.name == "frag_gen_test_channel" for r in caplog.records)


def test_unknown_total_omits_percentage_and_eta(caplog):
    """total=None (a streaming dataloader) reports counts without a bogus 0% / ETA."""
    with caplog.at_level(logging.INFO):
        progress = ProgressLogger(
            total=None, desc="Running inference", log_every=2, log_seconds=1e9
        )
        for _ in range(4):
            progress.update()
        progress.finish()

    messages = [r.message for r in caplog.records]
    headers = [m for m in messages if m.startswith("Running inference:")]
    assert headers == ["Running inference: 2/?", "Running inference: 4/?"]
    assert not [m for m in messages if "ETA" in m]
    assert "Total items processed: 4" in messages
