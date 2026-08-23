from __future__ import annotations

from pathlib import Path


def test_user_service_runs_only_the_locked_local_dashboard() -> None:
    repository = Path(__file__).resolve().parent.parent
    unit = repository / "systemd/netgrowth-paper-dashboard.service"

    assert not (repository / "systemd/netgrowth-paper-proof.service").exists()
    text = unit.read_text(encoding="utf-8")
    assert "Wants=network-online.target" in text
    assert "After=network-online.target" in text
    assert "WorkingDirectory=%h/Desktop/python-price-predict-experiment" in text
    assert (
        "ExecStart=%h/Desktop/python-price-predict-experiment/.venv/bin/netgrowth serve "
        "--device cuda --host 127.0.0.1 --port 8765"
    ) in text
    assert "Restart=on-failure" in text
    assert "WantedBy=default.target" in text
    assert "uv run" not in text
    assert "Forward Paper Proof" not in text
    assert "xdg-open" not in text
