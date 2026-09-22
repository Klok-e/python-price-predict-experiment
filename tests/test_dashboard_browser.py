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

TRADE_EVENTS = [
    {
        "id": "buy-1",
        "type": "InstrumentFilled",
        "time": "2026-08-23T18:16:00Z",
        "ticker": "BTCUSDT",
        "title": "Buy BTCUSDT",
        "trade": {"side": "buy", "quantity": 0.01, "price": 68_207.71, "cost": 0.48},
    },
    {
        "id": "sell-1",
        "type": "InstrumentFilled",
        "time": "2026-08-23T18:01:00Z",
        "ticker": "ETHUSDT",
        "title": "Sell ETHUSDT",
        "trade": {"side": "sell", "quantity": 0.2, "price": 2_998.0, "cost": 0.42},
    },
]


class _DashboardHandler(BaseHTTPRequestHandler):
    control_requests: list[tuple[str, dict[str, Any]]] = []
    response_overrides: dict[str, dict[str, Any]] = {}

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
                    "decision_blocker": None,
                    "pending_fill": None,
                    "recent_trades": TRADE_EVENTS,
                    "latest_decision": {
                        "event_id": "decision-1",
                        "decision_id": "policy-decision-1",
                        "signal_time": "2026-08-23T18:15:00Z",
                        "threshold_outcome": "below_threshold",
                        "execution_status": "not_required",
                    },
                    "hold_benchmark": {
                        "started_at": "2026-08-20T18:00:00Z",
                        "equity": 10_088.0,
                        "net_pnl": 88.0,
                        "compounded_net_return": 0.0088,
                        "maximum_drawdown": 0.012,
                        "funding": -2.0,
                        "gross_exposure": 0.5,
                        "excess_pnl": 17.25,
                        "excess_return": 0.001725,
                    },
                    "fitting": {
                        "status": "failed",
                        "attempt_count": 2,
                        "next_retry_at": "2026-08-23T18:35:00Z",
                        "last_successful_fit_at": "2026-08-20T18:00:00Z",
                        "error": "provider unavailable",
                    },
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
                    "accounts": [
                        {"id": "paper-7", "label": "Paper account 7", "active": True},
                        {"id": "paper-6", "label": "Paper account 6", "active": False},
                    ],
                    "selected_account_id": "paper-7",
                    "activity": {
                        "decisions": 18,
                        "executable_changes": 5,
                        "fills": 7,
                        "below_threshold": 9,
                        "unchanged_targets": 4,
                        "missed_executions": 1,
                        "interventions": 0,
                    },
                    "comparison": {
                        "account_id": "paper-7",
                        "protocol_id": "net-growth-v1",
                        "current_equity": 10_105.25,
                        "compounded_net_return": 0.010525,
                        "maximum_drawdown": 0.018,
                        "accounts": [
                            {
                                "account_id": "paper-7",
                                "protocol_id": "net-growth-v1",
                                "current_equity": 10_105.25,
                                "compounded_net_return": 0.010525,
                                "maximum_drawdown": 0.018,
                            },
                            {
                                "account_id": "paper-6",
                                "protocol_id": "net-growth-v0",
                                "current_equity": 9_850.0,
                                "compounded_net_return": -0.015,
                                "maximum_drawdown": 0.031,
                            },
                        ],
                    },
                    "protocol_segments": [
                        {
                            "protocol_id": "net-growth-v1",
                            "started_at": "2026-08-20T18:00:00Z",
                            "ended_at": None,
                            "compounded_net_return": 0.010525,
                            "decisions": 18,
                            "executable_changes": 5,
                            "hold_benchmark": {
                                "equity": 10_088.0,
                                "net_pnl": 88.0,
                                "compounded_net_return": 0.0088,
                                "maximum_drawdown": 0.012,
                                "funding": -2.0,
                                "gross_exposure": 0.5,
                                "excess_pnl": 17.25,
                                "excess_return": 0.001725,
                            },
                            "transaction_cost": 11.25,
                            "funding": -3.0,
                            "turnover": 22_400,
                            "gross_exposure": 0.5,
                            "below_threshold": 9,
                            "missed_executions": 1,
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
                        },
                        *TRADE_EVENTS,
                    ],
                }
            )
            return
        if path == "/api/system":
            self._json(
                {
                    "market_feed": {"status": "Fresh", "provider": "Binance USD-M"},
                    "policy": {
                        "protocol_id": "net-growth-v1",
                        "model_id": "fit-2026-08-18",
                        "fitted_at": "2026-08-20T18:00:00Z",
                        "age_seconds": 12_600,
                    },
                    "fitting": {
                        "status": "failed",
                        "attempt_count": 2,
                        "next_retry_at": "2026-08-23T18:35:00Z",
                        "last_successful_fit_at": "2026-08-20T18:00:00Z",
                        "error": "provider unavailable",
                    },
                    "revision": {"status": "candidate_ready", "candidate_protocol_id": "net-growth-v2"},
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
                            "hold_benchmark": {"equity": 10_085.0},
                            "drawdown": 0.003,
                            "gross_exposure": 0.45,
                            "net_exposure": 0.05,
                        },
                        {
                            "time": "2026-08-23T18:15:00Z",
                            "equity": 10_105.25,
                            "cash_benchmark": 10_000,
                            "equal_weight_benchmark": 10_082,
                            "hold_benchmark": {"equity": 10_088.0},
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
        if path in {"/api/events/buy-1", "/api/events/sell-1"}:
            trade = next(event for event in TRADE_EVENTS if path.endswith(event["id"]))
            self._json({**trade, "execution": {"outcome": "PortfolioChangeExecuted", "fills": [trade["trade"]]}})
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
                        "top_influences": [{"label": "BTC momentum / recent", "value": 0.31}],
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
        if isinstance(payload, dict):
            payload = {**payload, **self.response_overrides.get(self.path.partition("?")[0], {})}
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.fixture
def dashboard_server() -> Iterator[tuple[str, type[_DashboardHandler]]]:
    _DashboardHandler.control_requests = []
    _DashboardHandler.response_overrides = {}
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
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(base_url)

        playwright.expect(page.get_by_role("heading", name="Paper account")).to_be_visible()
        playwright.expect(page.get_by_text("$10,105.25", exact=True).first).to_be_visible()
        playwright.expect(page.get_by_text("Performance vs hold", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("$17.25", exact=True).first).to_be_visible()
        playwright.expect(page.get_by_text("Attempt 2", exact=False)).to_be_visible()
        playwright.expect(page.locator("#primary-metrics > *")).to_have_count(5)
        playwright.expect(page.locator("#financial-details")).not_to_have_attribute("open", "")
        playwright.expect(page.get_by_text("Starting equity", exact=True)).not_to_be_visible()
        playwright.expect(page.get_by_role("button", name="Resume", exact=True)).not_to_be_visible()
        playwright.expect(page.get_by_role("button", name="Reset account", exact=True)).not_to_be_visible()
        playwright.expect(page.locator("#latest-decision")).to_contain_text("below")
        page.locator("#latest-decision").get_by_role("button").click()
        playwright.expect(page.get_by_role("dialog", name="Event detail")).to_be_visible()
        page.get_by_role("button", name="Close event detail").click()

        chart = page.locator("#financial-chart")
        playwright.expect(
            chart.get_by_role("img", name="BTCUSDT synchronized paper account financial chart")
        ).to_be_visible()
        chart.get_by_role("button", name=re.compile(r"Signal marker.*BTC target increased")).click()
        detail = page.get_by_role("dialog", name="Event detail")
        playwright.expect(detail.get_by_role("heading", name="Decision record")).to_be_visible()
        playwright.expect(detail.get_by_role("heading", name="Execution outcome")).to_be_visible()
        detail.get_by_role("button", name="Close event detail").click()

        page.get_by_role("tab", name="History").click()
        playwright.expect(page.get_by_role("heading", name="Account history")).to_be_visible()
        playwright.expect(page.get_by_role("heading", name="Policy Protocol segments")).to_be_visible()
        playwright.expect(page.locator("#view-history").get_by_text("BTC target increased", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("Paper account 6", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("-1.50%", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("Hold 0.88%", exact=False)).to_be_visible()
        playwright.expect(page.get_by_text("Below threshold 9", exact=False)).to_be_visible()
        activity = page.locator("#history-activity")
        playwright.expect(activity).to_contain_text("18")
        original_activity = activity.text_content()
        page.locator("#event-type-select").select_option("Signal")
        playwright.expect(activity).to_have_text(original_activity)

        page.get_by_role("tab", name="System").click()
        playwright.expect(page.get_by_role("heading", name="System health")).to_be_visible()
        system = page.get_by_role("tabpanel", name="System")
        playwright.expect(system.get_by_text("net-growth-v1", exact=True)).to_be_visible()
        playwright.expect(system.get_by_text("Model age", exact=True)).to_be_visible()
        playwright.expect(system.get_by_text("Candidate Ready", exact=True)).to_be_visible()

        page.goto(f"{base_url}/history")
        playwright.expect(page.get_by_role("heading", name="Account history")).to_be_visible()

        page.get_by_role("tab", name="Live").click()
        page.get_by_role("button", name="Pause").click()
        confirmation = page.get_by_role("dialog", name="Confirm Pause")
        playwright.expect(confirmation.get_by_text("Review current exposure", exact=True)).to_be_visible()
        confirmation.get_by_role("button", name="Confirm Pause").click()
        playwright.expect(page.get_by_role("status")).to_contain_text("Control accepted")

        assert handler.control_requests == [("/api/controls/pause", {"expected_version": 12, "confirmation": "Pause"})]
        assert page_errors == []
        browser.close()


def test_executed_buys_and_sells_are_distinct_from_decisions(
    dashboard_server: tuple[str, type[_DashboardHandler]],
) -> None:
    base_url, _handler = dashboard_server
    with playwright.sync_playwright() as browser_tools:
        try:
            browser = browser_tools.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium is unavailable for the real-browser smoke: {error}")
        page = browser.new_page()
        page.goto(base_url)
        trades = page.locator("#recent-trades")
        playwright.expect(trades.get_by_role("button", name="Open simulated trade: Buy BTCUSDT")).to_be_visible()
        playwright.expect(trades).to_contain_text("0.01 at $68,207.71 · Cost $0.48")
        playwright.expect(trades).to_contain_text("Sell ETHUSDT")
        playwright.expect(trades).to_contain_text("0.2 at $2,998.00 · Cost $0.42")
        playwright.expect(trades).not_to_contain_text("BTC target increased")
        playwright.expect(page.locator("#latest-decision")).to_contain_text("No trade")
        trades.get_by_role("button", name="Open simulated trade: Buy BTCUSDT").click()
        detail = page.get_by_role("dialog", name="Event detail")
        playwright.expect(detail.get_by_role("heading", name="Buy BTCUSDT")).to_be_visible()
        playwright.expect(detail.get_by_role("heading", name="Execution outcome")).to_be_visible()
        detail.get_by_role("button", name="Close event detail").click()
        page.get_by_role("tab", name="History").click()
        page.locator("#event-type-select").select_option(label="Trades")
        history = page.locator("#history-events")
        playwright.expect(history.get_by_role("button")).to_have_count(2)
        playwright.expect(history).to_contain_text("Buy BTCUSDT")
        playwright.expect(history).to_contain_text("Sell ETHUSDT")
        playwright.expect(history).not_to_contain_text("BTC target increased")
        playwright.expect(page.locator("#history-count")).to_have_text("2 simulated trades")
        page.locator("#event-type-select").select_option("Signal")
        playwright.expect(history).to_contain_text("BTC target increased")
        playwright.expect(history).not_to_contain_text("Buy BTCUSDT")
        browser.close()


@pytest.mark.parametrize("event_type", ["ModelAttribution", "ModelAttributionFailed"])
def test_attribution_events_do_not_reintroduce_removed_section(
    dashboard_server: tuple[str, type[_DashboardHandler]], event_type: str
) -> None:
    base_url, handler = dashboard_server
    handler.response_overrides["/api/history"] = {
        "events": [{"id": "decision-1", "type": event_type, "title": "Model attribution result"}],
    }
    handler.response_overrides["/api/events/decision-1"] = {
        "type": event_type,
        "title": "Model attribution result",
        "details": {
            "label": "Approximate post-hoc influence evidence",
            "top_influences": [{"label": "BTC momentum", "value": 0.31}],
            "error": "Attribution computation failed" if event_type.endswith("Failed") else None,
        },
    }
    with playwright.sync_playwright() as browser_tools:
        try:
            browser = browser_tools.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium is unavailable for the real-browser smoke: {error}")
        page = browser.new_page()
        page.goto(f"{base_url}/history")
        page.locator("#event-type-select").select_option(event_type)
        page.get_by_role("button", name=f"Open {event_type} event: Model attribution result").click()
        detail = page.get_by_role("dialog", name="Event detail")
        playwright.expect(detail.get_by_role("heading", name="Decision record")).to_be_visible()
        playwright.expect(detail.get_by_role("heading", name="Execution outcome")).to_be_visible()
        playwright.expect(detail.get_by_role("heading", name="Event facts")).to_have_count(0)
        playwright.expect(detail).not_to_contain_text("Approximate post-hoc influence evidence")
        playwright.expect(detail).not_to_contain_text("BTC momentum")
        browser.close()


@pytest.mark.parametrize(
    ("lifecycle", "execution_status", "outcome", "blocker", "reason"),
    [
        ("Risk Stopped", "missed", "Execution missed", "account_not_trading", "Account is Risk stopped"),
        (
            "Migration Required",
            "cancelled",
            "Execution cancelled",
            "account_not_trading",
            "Account is Migration required",
        ),
        ("Trading", "executed", "Portfolio change executed", None, None),
        ("Trading", "executed", "Portfolio change executed", "policy_preparing", "Model preparation is in progress"),
        (
            "Trading",
            "executed",
            "Portfolio change executed",
            "revision_draining",
            "Policy revision is waiting for pending work to finish",
        ),
    ],
)
def test_dashboard_shows_decision_outcomes_and_blocking_states(
    dashboard_server: tuple[str, type[_DashboardHandler]],
    lifecycle: str,
    execution_status: str,
    outcome: str,
    blocker: str | None,
    reason: str | None,
) -> None:
    base_url, handler = dashboard_server
    handler.response_overrides["/api/live"] = {
        "account": {"id": "paper-7", "state": lifecycle, "version": 12, "current_equity": 10_105.25},
        "latest_decision": {
            "event_id": "decision-1",
            "decision_id": "policy-decision-1",
            "signal_time": "2026-08-23T18:15:00Z",
            "threshold_outcome": "executable",
            "execution_status": execution_status,
        },
        "controls": {"pause": lifecycle == "Trading", "resume": False, "flatten": False, "reset": False},
        "decision_blocker": blocker,
    }
    with playwright.sync_playwright() as browser_tools:
        try:
            browser = browser_tools.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium is unavailable for the real-browser smoke: {error}")
        page = browser.new_page()
        page.goto(base_url)
        playwright.expect(page.locator("#latest-decision")).to_contain_text(outcome)
        if blocker:
            playwright.expect(page.locator("#live-status")).to_contain_text("Suspended")
            playwright.expect(page.locator("#live-status")).to_contain_text(reason)
        if lifecycle != "Trading":
            playwright.expect(page.get_by_role("button", name="Resume", exact=True)).not_to_be_visible()
            playwright.expect(page.get_by_role("button", name="Pause", exact=True)).not_to_be_visible()
        browser.close()


def test_dashboard_refresh_preserves_disclosures_and_exposes_failures(
    dashboard_server: tuple[str, type[_DashboardHandler]],
) -> None:
    base_url, handler = dashboard_server
    handler.response_overrides["/api/history"] = {
        "comparison": {"protocol_id": "a" * 64, "current_equity": 10_105.25},
    }
    with playwright.sync_playwright() as browser_tools:
        try:
            browser = browser_tools.chromium.launch(headless=True)
        except playwright.Error as error:
            pytest.skip(f"Chromium is unavailable for the real-browser smoke: {error}")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.clock.install()
        page.goto(base_url)
        playwright.expect(page.locator("#primary-metrics")).to_contain_text("$10,105.25")
        playwright.expect(page.locator("#last-refresh")).to_contain_text("Updated")
        page.get_by_role("tab", name="History").click()
        playwright.expect(page.locator("#history-comparison")).to_contain_text("a" * 64)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.get_by_role("tab", name="Live").click()
        financial_summary = page.locator("#financial-details > summary")
        financial_summary.focus()
        page.keyboard.press("Enter")
        playwright.expect(page.get_by_text("Starting equity", exact=True)).to_be_visible()
        actions_summary = page.locator("#account-actions > summary")
        actions_summary.focus()
        page.keyboard.press("Enter")
        playwright.expect(page.get_by_role("button", name="Reset account", exact=True)).to_be_visible()

        handler.response_overrides["/api/live"] = {
            "account": {"id": "paper-7", "state": "Paused", "version": 12, "current_equity": None},
            "freshness": {"status": "Data Stale", "observed_at": None, "error": "Feed unavailable"},
            "hold_benchmark": None,
            "latest_decision": None,
            "fitting": {"status": "idle"},
            "decision_blocker": "account_not_trading",
            "recent_trades": [],
        }
        with page.expect_response("**/api/live"):
            page.clock.fast_forward(60_000)
        playwright.expect(page.get_by_role("button", name="Resume", exact=True)).to_be_visible()
        playwright.expect(page.get_by_role("button", name="Pause", exact=True)).not_to_be_visible()
        playwright.expect(page.locator("#live-status")).to_contain_text("Suspended")
        playwright.expect(page.locator("#live-status")).to_contain_text("Feed unavailable")
        playwright.expect(page.locator("#primary-metrics")).not_to_contain_text("$0.00")
        playwright.expect(page.locator("#recent-trades")).to_contain_text("No simulated trades have executed yet")
        playwright.expect(page.locator("#financial-details")).to_have_attribute("open", "")
        playwright.expect(page.locator("#account-actions")).to_have_attribute("open", "")
        playwright.expect(actions_summary).to_be_focused()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        page.route("**/api/live", lambda route: route.fulfill(status=503, body="unavailable"))
        with page.expect_response("**/api/live"):
            page.clock.fast_forward(60_000)
        playwright.expect(page.locator("#connection-banner")).to_be_visible()
        browser.close()
