from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

playwright = pytest.importorskip("playwright.sync_api", reason="Playwright is optional for the real-browser smoke")


STATIC_ROOT = Path(__file__).parents[1] / "netgrowth" / "paper_dashboard" / "static"


class _DashboardHandler(BaseHTTPRequestHandler):
    control_requests: list[tuple[str, dict[str, Any]]] = []

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.partition("?")[0]
        if path == "/api/live":
            self._json(
                {
                    "as_of": "2026-08-23T18:30:00Z",
                    "account": {
                        "id": "paper-7",
                        "state": "Trading",
                        "version": 12,
                        "started_at": "2026-08-20T18:00:00Z",
                        "starting_equity": 10_000,
                        "current_equity": 10_105.25,
                        "cash": 5_000,
                        "net_pnl": 105.25,
                        "compounded_net_return": 0.010525,
                        "gross_trading_pnl": 119.5,
                        "transaction_cost": 11.25,
                        "funding": -3.0,
                        "turnover": 22_400,
                    },
                    "risk": {
                        "current_drawdown": 0.004,
                        "maximum_drawdown": 0.018,
                        "high_water_equity": 10_145.83,
                        "drawdown_limit": 0.2,
                        "gross_exposure": 0.5,
                        "net_exposure": 0.1,
                        "cash_weight": 0.5,
                        "concentrations": {"BTCUSDT": 0.2},
                    },
                    "positions": [
                        {
                            "ticker": "BTCUSDT",
                            "side": "Long",
                            "quantity": 0.03,
                            "mark": 68_150,
                            "notional": 2_044.5,
                            "current_weight": 0.2,
                            "target_weight": 0.25,
                            "average_entry": 67_900,
                            "realized_pnl": 8.0,
                            "unrealized_pnl": 7.5,
                        }
                    ],
                    "activity": {
                        "decisions": 18,
                        "executable_changes": 5,
                        "fills": 7,
                        "below_threshold": 9,
                        "unchanged_targets": 4,
                        "missed_executions": 1,
                        "interventions": 0,
                    },
                    "freshness": {
                        "status": "Fresh",
                        "observed_at": "2026-08-23T18:29:54Z",
                        "age_seconds": 6,
                        "error": None,
                    },
                    "operating_window": {"started_at": "2026-08-23T16:02:00Z"},
                    "next_decision_at": "2026-08-23T18:45:00Z",
                    "pending_fill": None,
                    "fitting": {"status": "Idle", "next_fit_at": "2026-08-25T17:00:00Z"},
                    "recent_events": [
                        {
                            "id": "decision-1",
                            "type": "Signal",
                            "time": "2026-08-23T18:15:00Z",
                            "title": "BTC target increased",
                            "summary": "Executable Portfolio Change",
                        }
                    ],
                }
            )
            return
        if path == "/api/history":
            self._json(
                {
                    "accounts": [{"id": "paper-7", "label": "Paper account 7", "active": True}],
                    "selected_account_id": "paper-7",
                    "protocol_segments": [
                        {
                            "protocol_id": "net-growth-v1",
                            "started_at": "2026-08-20T18:00:00Z",
                            "ended_at": None,
                            "compounded_net_return": 0.010525,
                            "decisions": 18,
                            "executable_changes": 5,
                        }
                    ],
                    "events": [
                        {
                            "id": "decision-1",
                            "type": "Signal",
                            "time": "2026-08-23T18:15:00Z",
                            "title": "BTC target increased",
                            "summary": "Executable Portfolio Change",
                            "ticker": "BTCUSDT",
                        }
                    ],
                }
            )
            return
        if path == "/api/system":
            self._json(
                {
                    "market_feed": {"status": "Fresh", "provider": "Binance USD-M"},
                    "policy": {"protocol_id": "net-growth-v1", "model_id": "fit-2026-08-18"},
                    "fitting": {"status": "Idle", "device": "ROCm"},
                    "operating_windows": [{"started_at": "2026-08-23T16:02:00Z", "ended_at": None}],
                    "notifications": {"status": "Available", "last_delivery_at": None},
                    "database": {"status": "Healthy", "wal": True},
                    "backup": {"status": "Healthy", "last_backup_at": "2026-08-23T16:02:01Z", "retained": 7},
                    "service": {"status": "Running", "pid": 7001},
                }
            )
            return
        if path == "/api/csrf":
            self._json({"csrf_token": "browser-smoke-token"})
            return
        if path == "/api/chart":
            self._json(
                {
                    "ticker": "BTCUSDT",
                    "range": "current",
                    "candles": [
                        {
                            "time": "2026-08-23T18:14:00Z",
                            "open": 68_000,
                            "high": 68_120,
                            "low": 67_980,
                            "close": 68_100,
                        },
                        {
                            "time": "2026-08-23T18:15:00Z",
                            "open": 68_100,
                            "high": 68_220,
                            "low": 68_050,
                            "close": 68_180,
                        },
                    ],
                    "weights": [
                        {"time": "2026-08-23T18:14:00Z", "current": 0.15, "target": 0.15},
                        {"time": "2026-08-23T18:15:00Z", "current": 0.15, "target": 0.25},
                    ],
                    "portfolio": [
                        {
                            "time": "2026-08-23T18:14:00Z",
                            "equity": 10_100,
                            "cash_benchmark": 10_000,
                            "equal_weight_benchmark": 10_080,
                            "drawdown": 0.003,
                            "gross_exposure": 0.45,
                            "net_exposure": 0.05,
                        },
                        {
                            "time": "2026-08-23T18:15:00Z",
                            "equity": 10_105.25,
                            "cash_benchmark": 10_000,
                            "equal_weight_benchmark": 10_082,
                            "drawdown": 0.004,
                            "gross_exposure": 0.5,
                            "net_exposure": 0.1,
                        },
                    ],
                    "gaps": [],
                    "markers": [
                        {
                            "id": "decision-1",
                            "time": "2026-08-23T18:15:00Z",
                            "type": "Signal",
                            "label": "BTC target increased",
                            "price": 68_180,
                            "material": True,
                        }
                    ],
                }
            )
            return
        if path == "/api/events/decision-1":
            self._json(
                {
                    "id": "decision-1",
                    "type": "Signal",
                    "time": "2026-08-23T18:15:00Z",
                    "title": "BTC target increased",
                    "summary": "Executable Portfolio Change",
                    "ticker": "BTCUSDT",
                    "decision": {
                        "signal_time": "2026-08-23T18:15:00Z",
                        "model_id": "fit-2026-08-18",
                        "input_id": "market-state-991",
                        "current_portfolio": {"BTCUSDT": 0.15},
                        "raw_target_weights": {"BTCUSDT": 0.27},
                        "constrained_target_weights": {"BTCUSDT": 0.25},
                        "projected_turnover": 0.1,
                        "threshold_outcome": "Executable",
                        "fill_due_at": "2026-08-23T18:16:00Z",
                        "outcome": "Filled",
                    },
                    "execution": {"reference_price": 68_160, "effective_fill": 68_207.71, "cost": 1.43},
                    "attribution": {
                        "status": "Complete",
                        "label": "Approximate post-hoc influence evidence",
                        "top_influences": [{"feature": "BTC momentum / recent", "value": 0.31}],
                        "method": "Integrated Gradients",
                        "parameters": {"steps": 64},
                        "input_hash": "input-abc",
                        "model_hash": "model-def",
                    },
                }
            )
            return

        asset = "index.html" if path in {"/", "/live", "/history", "/system"} else path.lstrip("/")
        target = (STATIC_ROOT / asset).resolve()
        if not target.is_relative_to(STATIC_ROOT.resolve()) or not target.is_file():
            self.send_error(404)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
        }.get(target.suffix, "application/octet-stream")
        payload = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        self.control_requests.append((self.path, payload))
        self._json({"accepted": True, "message": "Control accepted"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, payload: object) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def dashboard_server() -> Iterator[tuple[str, type[_DashboardHandler]]]:
    _DashboardHandler.control_requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _DashboardHandler
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_dashboard_startup_navigation_marker_detail_and_control_wiring(
    dashboard_server: tuple[str, type[_DashboardHandler]],
) -> None:
    base_url, handler = dashboard_server

    with playwright.sync_playwright() as browser_tools:
        try:
            browser = browser_tools.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium is unavailable for the real-browser smoke: {error}")

        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(base_url)

        playwright.expect(page.get_by_role("heading", name="Paper account")).to_be_visible()
        playwright.expect(page.get_by_text("$10,105.25", exact=True).first).to_be_visible()

        page.get_by_role("button", name=re.compile(r"Signal.*BTC target increased")).click()
        detail = page.get_by_role("dialog", name="Event detail")
        playwright.expect(detail.get_by_role("heading", name="Decision record")).to_be_visible()
        playwright.expect(detail.get_by_text("Approximate post-hoc influence evidence", exact=True)).to_be_visible()
        detail.get_by_role("button", name="Close event detail").click()

        page.get_by_role("tab", name="History").click()
        playwright.expect(page.get_by_role("heading", name="Account history")).to_be_visible()
        playwright.expect(page.get_by_role("heading", name="Policy Protocol segments")).to_be_visible()
        playwright.expect(page.get_by_text("BTC target increased", exact=True).first).to_be_visible()

        page.get_by_role("tab", name="System").click()
        playwright.expect(page.get_by_role("heading", name="System health")).to_be_visible()
        playwright.expect(page.get_by_text("net-growth-v1", exact=True)).to_be_visible()

        page.goto(f"{base_url}/history")
        playwright.expect(page.get_by_role("heading", name="Account history")).to_be_visible()

        page.get_by_role("tab", name="Live").click()
        page.get_by_role("button", name="Pause").click()
        confirmation = page.get_by_role("dialog", name="Confirm Pause")
        playwright.expect(confirmation.get_by_text(re.compile("current exposure"))).to_be_visible()
        confirmation.get_by_role("button", name="Confirm Pause").click()
        playwright.expect(page.get_by_role("status")).to_contain_text("Control accepted")

        assert handler.control_requests == [("/api/controls/pause", {"expected_version": 12, "confirmation": "Pause"})]
        browser.close()
