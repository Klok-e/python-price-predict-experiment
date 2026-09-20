(function startDashboard() {
  "use strict";

  const API = {
    live: "/api/live",
    history: "/api/history",
    system: "/api/system",
    chart: "/api/chart",
    csrf: "/api/csrf",
    event: (eventId) => `/api/events/${encodeURIComponent(eventId)}`,
    control: (action) => `/api/controls/${encodeURIComponent(action)}`,
  };
  const KYIV_ZONE = "Europe/Kiev";
  const state = {
    live: null,
    history: null,
    system: null,
    chart: null,
    selectedTicker: "BTCUSDT",
    selectedRange: "current",
    selectedAccount: null,
    eventType: "all",
    activeView: "live",
    chartRenderer: null,
    csrfToken: null,
    toastTimer: null,
  };

  const $ = (id) => document.getElementById(id);
  const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const asArray = (value) => (Array.isArray(value) ? value : []);
  const first = (...values) => values.find((value) => value !== undefined && value !== null);
  const number = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  function element(tagName, options = {}, children = []) {
    const node = document.createElement(tagName);
    if (options.className) node.className = options.className;
    if (options.text !== undefined) node.textContent = String(options.text);
    for (const [key, value] of Object.entries(options.attributes ?? {})) {
      if (value !== undefined && value !== null) node.setAttribute(key, String(value));
    }
    node.append(...children.filter(Boolean));
    return node;
  }

  function unwrap(payload, key) {
    if (!isObject(payload)) return {};
    if (isObject(payload[key])) return payload[key];
    if (isObject(payload.data)) return payload.data;
    if (isObject(payload.snapshot)) return payload.snapshot;
    return payload;
  }

  function timestamp(value) {
    if (value instanceof Date) return value.getTime();
    if (typeof value === "number") return value < 1e12 ? value * 1000 : value;
    const parsed = Date.parse(String(value ?? ""));
    return Number.isFinite(parsed) ? parsed : null;
  }

  function formatMoney(value) {
    const parsed = number(value);
    if (parsed === null) return "—";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(parsed);
  }

  function formatNumber(value, maximumFractionDigits = 4) {
    const parsed = number(value);
    if (parsed === null) return "—";
    return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(parsed);
  }

  function formatPercent(value) {
    const parsed = number(value);
    return parsed === null ? "—" : `${(parsed * 100).toFixed(2)}%`;
  }

  function formatDateTime(value, options = {}) {
    const time = timestamp(value);
    if (time === null) return "—";
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: options.utc ? "UTC" : KYIV_ZONE,
      year: options.full ? "numeric" : undefined,
      month: options.full ? "short" : undefined,
      day: options.full ? "2-digit" : undefined,
      hour: "2-digit",
      minute: "2-digit",
      second: options.seconds ? "2-digit" : undefined,
      hourCycle: "h23",
      timeZoneName: options.zone ? "short" : undefined,
    }).format(new Date(time));
  }

  function timeNode(value, options = {}) {
    const time = timestamp(value);
    if (time === null) return element("span", { text: "—" });
    const kyiv = formatDateTime(time, { full: options.full, seconds: options.seconds, zone: true });
    const utc = formatDateTime(time, { utc: true, full: options.full, seconds: options.seconds, zone: true });
    return element("time", {
      text: options.showUtc ? `${kyiv} · ${utc}` : kyiv,
      attributes: { datetime: new Date(time).toISOString(), title: `${kyiv} · ${utc}` },
    });
  }

  function humanize(key) {
    const special = {
      id: "ID",
      pnl: "P&L",
      net_pnl: "Net P&L",
      gross_pnl: "Gross P&L",
      protocol_id: "Policy Protocol ID",
      model_id: "Fitted Policy ID",
      input_id: "Market State input ID",
      fitted_at: "Fitted at",
      age_seconds: "Model age",
      attempt_count: "Attempts",
      next_retry_at: "Next retry",
      last_successful_fit_at: "Last successful fit",
      utc: "UTC",
      wal: "SQLite WAL",
      pid: "Process ID",
    };
    if (special[key]) return special[key];
    return String(key)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function formatUnknown(value, key = "") {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (key === "age_seconds") return duration(value);
    if (key === "status" && typeof value === "string") return humanize(value);
    if (Array.isArray(value)) return value.map((item) => formatUnknown(item)).join(", ") || "—";
    if (isObject(value)) {
      return Object.entries(value)
        .map(([nestedKey, nestedValue]) => `${humanize(nestedKey)}: ${formatUnknown(nestedValue, nestedKey)}`)
        .join(" · ");
    }
    if (typeof value === "number") return formatNumber(value);
    if (/(_at|_time|timestamp|started|ended|deadline)$/.test(key) && timestamp(value) !== null) {
      return `${formatDateTime(value, { full: true, seconds: true, zone: true })} · ${formatDateTime(value, { utc: true, full: true, seconds: true, zone: true })}`;
    }
    return String(value);
  }

  function duration(seconds) {
    const parsed = Math.max(0, Math.round(number(seconds) ?? 0));
    if (parsed < 60) return `${parsed}s`;
    if (parsed < 3600) return `${Math.floor(parsed / 60)}m ${parsed % 60}s`;
    const hours = Math.floor(parsed / 3600);
    const minutes = Math.floor((parsed % 3600) / 60);
    if (hours < 24) return `${hours}h ${minutes}m`;
    return `${Math.floor(hours / 24)}d ${hours % 24}h`;
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    const contentType = response.headers.get("content-type") ?? "";
    const responseText = await response.text();
    let payload = responseText;
    if (contentType.includes("application/json") || /^[\[{]/.test(responseText.trim())) {
      try {
        payload = responseText ? JSON.parse(responseText) : {};
      } catch {
        payload = responseText;
      }
    }
    if (!response.ok) {
      const message = isObject(payload) ? first(payload.detail, payload.message, response.statusText) : payload;
      throw new Error(String(message || `HTTP ${response.status}`));
    }
    return payload;
  }

  function setConnectionError(error) {
    const banner = $("connection-banner");
    banner.textContent = `The local service did not return a current dashboard snapshot: ${error.message}`;
    banner.hidden = false;
    $("freshness-pill").textContent = "Unavailable";
    $("freshness-pill").className = "status-pill status-danger";
  }

  function clearConnectionError() {
    $("connection-banner").hidden = true;
  }

  function showToast(message) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.add("toast-visible");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => toast.classList.remove("toast-visible"), 6000);
  }

  function metricCard(label, value, detail = "", sentiment = null) {
    const valueClass = sentiment === "positive" ? "value value-positive" : sentiment === "negative" ? "value value-negative" : "value";
    return element("article", { className: "metric-card" }, [
      element("span", { className: "label", text: label }),
      element("strong", { className: valueClass, text: value }),
      detail ? element("span", { className: "detail", text: detail }) : null,
    ]);
  }

  function heroCard(label, value, detail = "") {
    return element("article", { className: "hero-card" }, [
      element("span", { className: "label", text: label }),
      element("strong", { className: "value", text: value }),
      element("span", { className: "detail", text: detail }),
    ]);
  }

  function sentiment(value, inverted = false) {
    const parsed = number(value);
    if (parsed === null || parsed === 0) return null;
    const positive = inverted ? parsed < 0 : parsed > 0;
    return positive ? "positive" : "negative";
  }

  function liveParts() {
    const live = state.live ?? {};
    return {
      live,
      account: first(live.account, live.paper_account, {}),
      risk: first(live.risk, live.risk_metrics, {}),
      freshness: first(live.freshness, live.market_feed, live.data_status, {}),
      activity: first(live.activity, live.activity_metrics, {}),
      fitting: first(live.fitting, live.fit, {}),
    };
  }

  async function loadLive() {
    const payload = await fetchJson(API.live);
    state.live = unwrap(payload, "live");
    renderLive();
    clearConnectionError();
    return state.live;
  }

  async function loadHistory(accountId = state.selectedAccount) {
    const query = accountId ? `?account_id=${encodeURIComponent(accountId)}` : "";
    const payload = await fetchJson(`${API.history}${query}`);
    state.history = unwrap(payload, "history");
    renderHistory();
  }

  async function loadSystem() {
    const payload = await fetchJson(API.system);
    state.system = unwrap(payload, "system");
    renderSystem();
  }

  async function loadChart() {
    const pixelWidth = Math.max(320, Math.round($("financial-chart").clientWidth || 900));
    const query = new URLSearchParams({
      ticker: state.selectedTicker,
      range: state.selectedRange,
      pixels: String(pixelWidth),
    });
    $("financial-chart").setAttribute("aria-busy", "true");
    const payload = await fetchJson(`${API.chart}?${query}`);
    state.chart = unwrap(payload, "chart");
    if (state.chart.ticker) state.selectedTicker = state.chart.ticker;
    renderTickerOptions();
    renderChartLegend();
    state.chartRenderer.update(state.chart);
  }

  function renderChartLegend() {
    const legend = document.querySelector(".chart-legend");
    if (!legend || legend.querySelector("[data-legend='hold-benchmark']")) return;
    const portfolio = asArray(first(state.chart?.portfolio, state.chart?.account, state.chart?.panels));
    if (!portfolio.some((point) => isObject(point.hold_benchmark) || number(point.hold_benchmark_equity) !== null)) return;
    legend.append(element("span", { attributes: { "data-legend": "hold-benchmark" } }, [
      element("i", { className: "legend-current" }),
      document.createTextNode("Hold benchmark"),
    ]));
  }

  async function loadCsrf() {
    const payload = await fetchJson(API.csrf);
    state.csrfToken = first(payload.csrf_token, payload.token, payload.csrf);
    if (!state.csrfToken) throw new Error("The local control API did not provide a CSRF token");
    return state.csrfToken;
  }

  async function refreshAll() {
    try {
      await loadLive();
      const tasks = [loadHistory(), loadSystem(), loadChart()];
      const results = await Promise.allSettled(tasks);
      const rejected = results.find((result) => result.status === "rejected");
      if (rejected) throw rejected.reason;
      $("last-refresh").textContent = `Last refreshed ${formatDateTime(Date.now(), { seconds: true, zone: true })} · ${formatDateTime(Date.now(), { utc: true, seconds: true, zone: true })}`;
    } catch (error) {
      setConnectionError(error instanceof Error ? error : new Error(String(error)));
    }
  }

  function renderLive() {
    const { live, account, risk, freshness, activity, fitting } = liveParts();
    const accountId = first(account.id, account.account_id, live.account_id, "Unidentified account");
    const lifecycle = first(account.state, account.lifecycle_state, live.state, "Unknown");
    const asOf = first(live.as_of, live.observed_at, freshness.observed_at);
    $("live-subtitle").textContent = `${accountId} · ${lifecycle} · snapshot ${formatDateTime(asOf, { full: true, seconds: true, zone: true })}`;

    const status = String(first(freshness.status, freshness.state, "Unknown"));
    const statusLower = status.toLowerCase();
    const freshnessPill = $("freshness-pill");
    freshnessPill.textContent = status;
    freshnessPill.className = `status-pill ${statusLower.includes("fresh") && !statusLower.includes("stale") ? "status-good" : statusLower.includes("stale") || statusLower.includes("error") ? "status-danger" : "status-warning"}`;

    const operatingWindow = first(live.operating_window, live.current_operating_window, {});
    const nextDecision = first(live.next_decision_at, live.next_signal_at, account.next_decision_at);
    const pendingFill = first(live.pending_fill, live.pending_change, account.pending_fill);
    const observedAt = first(freshness.observed_at, freshness.last_observation_at, live.as_of);
    const ageSeconds = first(freshness.age_seconds, observedAt ? (Date.now() - timestamp(observedAt)) / 1000 : null);
    const fittingStatus = first(fitting.status, fitting.state, "Unknown");
    const overview = [
      heroCard("Lifecycle", lifecycle, `Account ${accountId} · version ${first(account.version, live.version, "—")}`),
      heroCard(
        "Operating window",
        operatingWindow.started_at ? "Active" : first(operatingWindow.status, "Not reported"),
        operatingWindow.started_at ? `Started ${formatDateTime(operatingWindow.started_at, { full: true, zone: true })}` : formatUnknown(operatingWindow),
      ),
      heroCard("Market data", status, `${duration(ageSeconds)} old · ${freshness.error || freshness.last_error || "no active error"}`),
      heroCard("Next decision", nextDecision ? formatDateTime(nextDecision, { seconds: true, zone: true }) : "Not scheduled", nextDecision ? `Due in ${duration((timestamp(nextDecision) - Date.now()) / 1000)}` : "No decision countdown"),
      heroCard(
        "Pending fill",
        pendingFill ? first(pendingFill.status, pendingFill.outcome, "Pending") : "None",
        pendingFill ? formatUnknown(pendingFill) : "No Portfolio Change awaiting execution",
      ),
      heroCard(
        "Policy fitting",
        humanize(fittingStatus),
        fittingDetail(fitting),
      ),
    ];
    $("live-overview").replaceChildren(...overview);

    const accountMetrics = [
      ["Starting equity", formatMoney(first(account.starting_equity, account.initial_equity))],
      ["Current Marked Equity", formatMoney(first(account.current_equity, account.marked_equity, live.marked_equity))],
      ["Cash", formatMoney(first(account.cash, account.cash_balance))],
      ["Net P&L", formatMoney(first(account.net_pnl, account.pnl)), "Reconciles to Marked Equity", sentiment(first(account.net_pnl, account.pnl))],
      ["Compounded Net Return", formatPercent(first(account.compounded_net_return, account.net_return)), "Development evidence only", sentiment(first(account.compounded_net_return, account.net_return))],
      ["Gross trading P&L", formatMoney(first(account.gross_trading_pnl, account.gross_pnl)), "Before cost and funding", sentiment(first(account.gross_trading_pnl, account.gross_pnl))],
      ["Transaction Cost", formatMoney(first(account.transaction_cost, account.transaction_costs, account.costs)), "All-in adverse execution cost", "negative"],
      ["Funding paid / received", formatMoney(first(account.funding, account.funding_pnl, account.funding_cashflow)), "Separate from Transaction Cost", sentiment(first(account.funding, account.funding_pnl, account.funding_cashflow))],
      ["Turnover", formatMoney(first(account.turnover, account.total_turnover)), "Executed Portfolio Changes"],
      ["Composed equity", formatMoney(first(account.marked_equity_reconciliation?.composed_equity, account.composed_equity)), "Cash balance + unrealized P&L"],
      ["Reconciliation difference", formatMoney(first(account.marked_equity_reconciliation?.difference, account.reconciliation_difference)), "Composed equity + difference = authoritative Marked Equity", sentiment(first(account.marked_equity_reconciliation?.difference, account.reconciliation_difference), true)],
    ];
    const hold = first(live.hold_benchmark, account.hold_benchmark);
    if (isObject(hold)) {
      accountMetrics.push(
        ["Hold benchmark", formatMoney(hold.equity), "Positions held from the revision start"],
        ["Account vs hold", formatMoney(hold.excess_pnl), `${formatPercent(hold.excess_return)} difference`, sentiment(hold.excess_pnl)],
        ["Hold return", formatPercent(hold.compounded_net_return), `Maximum drawdown ${formatPercent(hold.maximum_drawdown)}`, sentiment(hold.compounded_net_return)],
        ["Hold funding", formatMoney(hold.funding), `Gross exposure ${formatPercent(hold.gross_exposure)}`, sentiment(hold.funding)],
      );
    }
    $("account-metrics").replaceChildren(...accountMetrics.map((metric) => metricCard(...metric)));

    const concentration = first(risk.concentrations, risk.per_instrument_concentration, {});
    const concentrationText = isObject(concentration)
      ? Object.entries(concentration).map(([ticker, value]) => `${ticker} ${formatPercent(value)}`).join(" · ") || "—"
      : formatUnknown(concentration);
    const riskMetrics = [
      ["Current drawdown", formatPercent(first(risk.current_drawdown, risk.drawdown)), "From high-water equity", sentiment(first(risk.current_drawdown, risk.drawdown), true)],
      ["Maximum Drawdown", formatPercent(first(risk.maximum_drawdown, risk.max_drawdown)), "Account lifetime", sentiment(first(risk.maximum_drawdown, risk.max_drawdown), true)],
      ["High-water equity", formatMoney(first(risk.high_water_equity, risk.equity_high_water))],
      ["Drawdown Limit", formatPercent(first(risk.drawdown_limit, risk.max_drawdown_limit, 0.2)), "Risk Stop at breach"],
      ["Gross exposure", formatPercent(first(risk.gross_exposure, account.gross_exposure))],
      ["Net exposure", formatPercent(first(risk.net_exposure, account.net_exposure))],
      ["Cash Weight", formatPercent(first(risk.cash_weight, account.cash_weight))],
      ["Concentration", concentrationText, "Per-instrument Current Weight"],
    ];
    $("risk-metrics").replaceChildren(...riskMetrics.map((metric) => metricCard(...metric)));

    renderPositions(asArray(first(live.positions, account.positions)));
    renderActivity(activity);
    renderRecentEvents(asArray(first(live.recent_events, live.events, live.recent_activity)));
    renderTickerOptions();
    updateControlAvailability(lifecycle, risk);
  }

  function fittingDetail(fitting) {
    const details = [];
    if (fitting.next_retry_at) details.push(`Retry ${formatDateTime(fitting.next_retry_at, { full: true, zone: true })}`);
    if (fitting.attempt_count !== undefined) details.push(`Attempt ${formatNumber(fitting.attempt_count, 0)}`);
    if (fitting.last_successful_fit_at) details.push(`Last successful fit ${formatDateTime(fitting.last_successful_fit_at, { full: true, zone: true })}`);
    if (fitting.error || fitting.last_error) details.push(String(first(fitting.error, fitting.last_error)));
    if (!details.length && fitting.progress !== undefined) details.push(`${formatPercent(fitting.progress)} complete`);
    if (!details.length && fitting.next_fit_at) details.push(`Next ${formatDateTime(fitting.next_fit_at, { full: true, zone: true })}`);
    return details.join(" · ") || "No fitting detail";
  }

  function renderPositions(positions) {
    const body = $("positions-body");
    if (!positions.length) {
      body.replaceChildren(element("tr", {}, [element("td", { className: "empty-cell", text: "Flat portfolio — no open instrument exposure.", attributes: { colspan: 10 } })]));
      $("positions-summary").textContent = "0 instruments with open exposure";
      return;
    }
    const rows = positions.map((position) => {
      const side = String(first(position.side, number(position.quantity) > 0 ? "Long" : number(position.quantity) < 0 ? "Short" : "Flat"));
      const cells = [
        [first(position.ticker, position.symbol, position.instrument, "—"), ""],
        [side, `side-${side.toLowerCase()}`],
        [formatNumber(position.quantity, 8), "numeric"],
        [formatMoney(first(position.mark, position.mark_price)), "numeric"],
        [formatMoney(position.notional), "numeric"],
        [formatPercent(first(position.current_weight, position.weight)), "numeric"],
        [formatPercent(first(position.target_weight, position.target)), "numeric"],
        [formatMoney(first(position.average_entry, position.average_entry_price, position.avg_entry)), "numeric"],
        [formatMoney(first(position.realized_pnl, position.realized)), "numeric"],
        [formatMoney(first(position.unrealized_pnl, position.unrealized)), "numeric"],
      ];
      return element("tr", {}, cells.map(([value, className]) => element("td", { className, text: value })));
    });
    body.replaceChildren(...rows);
    $("positions-summary").textContent = `${positions.length} instrument${positions.length === 1 ? "" : "s"}`;
  }

  function renderActivity(activity) {
    const metrics = [
      ["All decisions", first(activity.decisions, activity.decision_count)],
      ["Executable changes", first(activity.executable_changes, activity.executable_portfolio_changes)],
      ["Instrument fills", first(activity.fills, activity.instrument_fills)],
      ["Below threshold", first(activity.below_threshold, activity.below_threshold_decisions)],
      ["Unchanged targets", first(activity.unchanged_targets, activity.no_change_decisions)],
      ["Missed Executions", first(activity.missed_executions, activity.missed_execution_count)],
      ["Operator interventions", first(activity.interventions, activity.operator_interventions)],
    ];
    $("activity-metrics").replaceChildren(
      ...metrics.map(([label, value]) => element("div", { className: "activity-item" }, [
        element("span", { text: label }),
        element("strong", { text: formatNumber(value, 0) }),
      ])),
    );
  }

  function renderRecentEvents(events) {
    const list = $("recent-activity");
    if (!events.length) {
      list.replaceChildren(element("li", { className: "empty-state", text: "No material account events are retained yet." }));
      return;
    }
    list.replaceChildren(...events.slice(0, 10).map((event) => eventListItem(event)));
  }

  function eventListItem(event) {
    const eventId = first(event.id, event.event_id);
    const type = first(event.type, event.event_type, "Event");
    const title = first(event.title, event.label, humanize(type));
    const button = element("button", {
      className: "event-button",
      attributes: { type: "button", "aria-label": `Open ${type} event: ${title}` },
    }, [
      timeNode(first(event.time, event.timestamp, event.at), { seconds: false }),
      element("span", {}, [
        element("span", { className: "event-title", text: title }),
        element("span", { className: "event-summary", text: first(event.summary, event.outcome, type) }),
      ]),
    ]);
    button.addEventListener("click", () => openEvent(eventId));
    return element("li", {}, [button]);
  }

  function renderTickerOptions() {
    const select = $("ticker-select");
    const live = state.live ?? {};
    const positionTickers = asArray(first(live.positions, live.account?.positions)).map((position) => first(position.ticker, position.symbol, position.instrument));
    const universeValues = first(live.trading_universe, live.tickers, live.universe, []);
    const universe = Array.isArray(universeValues) ? universeValues.map((item) => (isObject(item) ? first(item.ticker, item.symbol) : item)) : Object.keys(universeValues ?? {});
    const chartTicker = first(state.chart?.ticker, state.selectedTicker);
    const tickers = [...new Set([chartTicker, ...universe, ...positionTickers].filter(Boolean))];
    if (!tickers.length) tickers.push(state.selectedTicker);
    if (!tickers.includes(state.selectedTicker)) state.selectedTicker = tickers[0];
    select.replaceChildren(...tickers.map((ticker) => element("option", { text: ticker, attributes: { value: ticker } })));
    select.value = state.selectedTicker;
  }

  function updateControlAvailability(lifecycle, risk) {
    const normalized = String(lifecycle).toLowerCase();
    const grossExposure = number(first(risk.gross_exposure, state.live?.account?.gross_exposure)) ?? 0;
    const available = first(state.live?.controls, state.live?.control_availability, {});
    const buttons = Object.fromEntries([...document.querySelectorAll("[data-control]")].map((button) => [button.dataset.control, button]));
    buttons.pause.disabled = available.pause !== undefined ? !available.pause : !normalized.includes("trading");
    buttons.resume.disabled = available.resume !== undefined
      ? !available.resume
      : !normalized.includes("paused") || normalized.includes("risk") || normalized.includes("reset");
    buttons.flatten.disabled = available.flatten !== undefined
      ? !available.flatten
      : grossExposure === 0 || normalized.includes("reset pending") || normalized.includes("migration");
    buttons.reset.disabled = available.reset !== undefined
      ? !available.reset
      : normalized.includes("reset pending") || normalized.includes("migration");
  }

  function renderHistory() {
    const history = state.history ?? {};
    const accounts = asArray(first(history.accounts, history.paper_accounts, history.archived_accounts));
    const selectedFromApi = first(history.selected_account_id, history.account_id);
    state.selectedAccount = first(state.selectedAccount, selectedFromApi, accounts.find((account) => account.active)?.id, accounts[0]?.id);
    const accountSelect = $("account-select");
    accountSelect.replaceChildren(...accounts.map((account) => {
      const accountId = first(account.id, account.account_id);
      const label = first(account.label, account.name, accountId);
      return element("option", { text: `${label}${account.active ? " · active" : " · archived"}`, attributes: { value: accountId } });
    }));
    if (state.selectedAccount) accountSelect.value = state.selectedAccount;

    const events = asArray(first(history.events, history.timeline, history.items));
    const eventTypes = [...new Set(events.map((event) => first(event.type, event.event_type, "Event")))].sort();
    const eventTypeSelect = $("event-type-select");
    const availableValues = [...eventTypeSelect.options].map((option) => option.value);
    if (eventTypes.some((type) => !availableValues.includes(type))) {
      eventTypeSelect.replaceChildren(
        element("option", { text: "All material events", attributes: { value: "all" } }),
        ...eventTypes.map((type) => element("option", { text: humanize(type), attributes: { value: type } })),
      );
    }
    eventTypeSelect.value = eventTypes.includes(state.eventType) ? state.eventType : "all";
    state.eventType = eventTypeSelect.value;
    const filtered = state.eventType === "all" ? events : events.filter((event) => first(event.type, event.event_type, "Event") === state.eventType);
    $("history-count").textContent = `${filtered.length} retained event${filtered.length === 1 ? "" : "s"}`;
    $("history-events").replaceChildren(...(filtered.length ? filtered.map((event) => timelineItem(event)) : [element("li", { className: "empty-state", text: "No material events match this account and filter." })]));
    renderHistoryComparison(history, accounts);
    renderProtocolSegments(history);
  }

  function timelineItem(event) {
    const eventId = first(event.id, event.event_id);
    const type = first(event.type, event.event_type, "Event");
    const title = first(event.title, event.label, humanize(type));
    const button = element("button", {
      className: "timeline-event",
      attributes: { type: "button", "aria-label": `Open ${type} event: ${title}` },
    }, [
      element("span", { className: "timeline-event-topline" }, [
        element("strong", { className: "event-title", text: title }),
        element("span", { className: "event-type", text: humanize(type) }),
      ]),
      element("span", { className: "event-summary", text: first(event.summary, event.outcome, event.ticker, "Open durable detail") }),
    ]);
    button.addEventListener("click", () => openEvent(eventId));
    const time = first(event.time, event.timestamp, event.at);
    return element("li", { className: "timeline-item" }, [
      element("div", { className: "timeline-time" }, [timeNode(time, { full: true, seconds: true }), element("br"), element("span", { text: `${formatDateTime(time, { utc: true, seconds: true })} UTC` })]),
      button,
    ]);
  }

  function renderHistoryComparison(history, accounts) {
    const selected = accounts.find((account) => account.id === state.selectedAccount) ?? {};
    const comparison = history.comparison ?? {};
    const metrics = [
      ["Account", selected.label ?? state.selectedAccount ?? "—"],
      ["Lifecycle", selected.active ? "Active" : "Archived"],
      ["Marked Equity", formatMoney(comparison.current_equity)],
      ["Compounded Net Return", formatPercent(comparison.compounded_net_return)],
      ["Maximum Drawdown", formatPercent(comparison.maximum_drawdown)],
      ["Policy Protocol", comparison.protocol_id ?? "—"],
      ["Started", formatDateTime(selected.started_at, { full: true, zone: true })],
      ["Archived", formatDateTime(selected.archived_at, { full: true, zone: true })],
    ];
    const accountComparisons = asArray(comparison.accounts).map((account) => {
      const accountId = account.account_id;
      const accountRow = accounts.find((candidate) => candidate.id === accountId);
      const label = accountRow?.label ?? accountId;
      const detail = [
        `Equity ${formatMoney(account.current_equity)}`,
        `Max drawdown ${formatPercent(account.maximum_drawdown)}`,
        `Protocol ${account.protocol_id}`,
      ].join(" · ");
      return metricCard(
        label,
        formatPercent(account.compounded_net_return),
        `${accountId === state.selectedAccount ? "Selected · " : ""}${detail}`,
      );
    });
    $("history-comparison").replaceChildren(
      ...metrics.map((metric) => metricCard(...metric)),
      ...accountComparisons,
    );
  }

  function renderProtocolSegments(history) {
    const segments = asArray(first(history.protocol_segments, history.policy_protocol_segments));
    const cards = segments.map((segment) => {
      const protocol = String(first(segment.protocol_id, "unknown"));
      const ended = first(segment.ended_at, segment.end);
      const interval = `${formatDateTime(first(segment.started_at, segment.start), { full: true })} – ${ended ? formatDateTime(ended, { full: true }) : "active"}`;
      const activity = `${first(segment.decisions, 0)} decisions · ${first(segment.executable_changes, 0)} executable changes`;
      const hold = isObject(segment.hold_benchmark) ? segment.hold_benchmark : null;
      const performance = hold
        ? `Account ${formatPercent(first(segment.compounded_net_return, segment.net_return, 0))} · Hold ${formatPercent(hold.compounded_net_return)}`
        : `${formatPercent(first(segment.compounded_net_return, segment.net_return, 0))} · ${activity}`;
      const details = [activity, interval];
      if (hold) {
        details.unshift(`Difference ${formatMoney(hold.excess_pnl)} · ${formatPercent(hold.excess_return)}`);
        details.push(`Hold equity ${formatMoney(hold.equity)} · hold drawdown ${formatPercent(hold.maximum_drawdown)}`);
      }
      details.push(
        `Cost ${formatMoney(segment.transaction_cost)} · funding ${formatMoney(segment.funding)} · turnover ${formatMoney(segment.turnover)}`,
        `Gross exposure ${formatPercent(segment.gross_exposure)} · Below threshold ${formatNumber(segment.below_threshold, 0)} · missed executions ${formatNumber(segment.missed_executions, 0)}`,
      );
      return metricCard(
        `Protocol ${protocol.slice(0, 12)}`,
        performance,
        details.join(" · "),
      );
    });
    $("history-protocol-segments").replaceChildren(
      ...(cards.length ? cards : [metricCard("No protocol segment", "Waiting for the first durable Policy Revision boundary")]),
    );
  }

  function renderSystem() {
    const system = state.system ?? {};
    const accepted = [
      ["Market feed", first(system.market_feed, system.freshness, system.data)],
      ["Policy and model", first(system.policy, system.model, system.policy_protocol)],
      ["Fitting", first(system.fitting, system.fit)],
      ["Compute runtime", first(system.compute, system.runtime, system.accelerator)],
      ["Operating windows", first(system.operating_windows, system.windows)],
      ["Notifications", first(system.notifications, system.notification_health)],
      ["Database", first(system.database, system.persistence)],
      ["Backup", first(system.backup, system.backups)],
      ["Service", first(system.service, system.diagnostics)],
    ].filter(([, value]) => value !== undefined && value !== null);
    const knownValues = new Set(accepted.map(([, value]) => value));
    const extras = Object.entries(system).filter(([key, value]) => !knownValues.has(value) && value !== undefined && value !== null && !["as_of", "timestamp"].includes(key));
    const sections = [...accepted, ...extras.map(([key, value]) => [humanize(key), value])];
    $("system-sections").replaceChildren(...sections.map(([title, value]) => systemCard(title, value)));
  }

  function systemCard(title, value) {
    return element("section", { className: "system-card" }, [
      element("h2", { text: title }),
      definitionList(value),
    ]);
  }

  function definitionList(value) {
    const list = element("dl", { className: "definition-list" });
    const entries = isObject(value)
      ? Object.entries(value)
      : Array.isArray(value)
        ? value.map((item, index) => [`${index + 1}`, item])
        : [["Status", value]];
    if (!entries.length) entries.push(["Status", "No detail reported"]);
    for (const [key, entryValue] of entries) {
      list.append(
        element("dt", { text: humanize(key) }),
        element("dd", { text: formatUnknown(entryValue, key) }),
      );
    }
    return list;
  }

  async function openEvent(eventId) {
    if (!eventId) {
      showToast("This chart item does not expose a durable event ID.");
      return;
    }
    const dialog = $("event-dialog");
    const detail = $("event-detail");
    detail.replaceChildren(element("p", { className: "empty-state", text: "Loading durable event detail…" }));
    if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
    else dialog.setAttribute("open", "");
    try {
      const payload = await fetchJson(API.event(eventId));
      renderEventDetail(unwrap(payload, "event"));
    } catch (error) {
      detail.replaceChildren(element("p", { className: "empty-state", text: `Event detail is unavailable: ${error.message}` }));
    }
  }

  function renderEventDetail(event) {
    const detail = $("event-detail");
    const title = event.title ?? humanize(event.type);
    const header = eventDetailSection("Event", [
      element("span", { className: "detail-badge", text: humanize(event.type) }),
      element("h3", { text: title }),
      element("p", { className: "detail-intro", text: event.summary ?? "Durable Paper Account history" }),
      definitionList({
        event_id: event.id,
        ticker: event.ticker,
        Kyiv: formatDateTime(event.time, { full: true, seconds: true, zone: true }),
        UTC: formatDateTime(event.time, { utc: true, full: true, seconds: true, zone: true }),
        decision_id: event.decision_id,
      }),
    ]);
    const sections = [header];
    const details = event.details;
    if (details && Object.keys(details).length && event.type !== "DecisionRecord") {
      sections.push(eventDetailSection("Event facts", [definitionList(details)]));
    }
    const decision = event.decision;
    if (decision) {
      sections.push(eventDetailSection("Decision record", [
        element("span", { className: "detail-badge", text: "Exact authoritative record" }),
        element("p", { className: "detail-intro", text: "Persisted Market State, Current Portfolio, Target Weights, constraints, timing, and final policy outcome." }),
        definitionList(decision),
      ]));
    }
    const execution = event.execution;
    if (execution) {
      sections.push(eventDetailSection("Execution outcome", [
        element("span", { className: "detail-badge", text: "Exact account mutation" }),
        definitionList(execution),
      ]));
    }
    const attribution = event.attribution;
    if (attribution) sections.push(attributionSection(attribution));
    else if (decision) {
      sections.push(eventDetailSection("Model attribution", [
        element("span", { className: "detail-badge detail-badge-approximate", text: "Attribution pending" }),
        element("p", { className: "detail-intro", text: "Approximate post-hoc influence evidence has not completed. It cannot delay execution." }),
      ]));
    }
    detail.replaceChildren(...sections);
  }

  function eventDetailSection(title, children) {
    return element("section", { className: "event-detail-section" }, [element("h3", { text: title }), ...children]);
  }

  function attributionSection(attribution) {
    const label = attribution.label ?? "Approximate post-hoc influence evidence";
    const influences = asArray(attribution.top_influences);
    const influenceList = element("ol", { className: "influence-list" }, influences.map((influence) => element("li", {}, [
      element("span", { text: influence.label ?? "Influence" }),
      element("strong", { text: formatNumber(influence.value, 5) }),
    ])));
    return eventDetailSection("Model attribution", [
      element("span", { className: "detail-badge detail-badge-approximate", text: label }),
      element("p", { className: "detail-intro", text: "Influence evidence is approximate and post-hoc; it is not a causal explanation, trade reason, or profitability claim." }),
      influences.length ? influenceList : element("p", { className: "detail-intro", text: attribution.status ?? "No top influences reported" }),
      definitionList({
        status: attribution.status,
        method: attribution.method,
        parameters: attribution.parameters,
        input_hash: attribution.input_hash,
        model_hash: attribution.model_hash,
      }),
    ]);
  }

  const controls = {
    pause: {
      label: "Pause",
      description: "Pause stops new policy decisions and pending policy fills while retaining current exposure. It does not flatten the account.",
    },
    resume: {
      label: "Resume",
      description: "Resume permits decisions from the next naturally due closed Decision Bar. It never replays decisions missed while paused.",
    },
    flatten: {
      label: "Flatten and pause",
      confirmation: "FLATTEN",
      description: "Flatten and pause cancels pending policy work and targets all exposure flat at the next fresh quote, with normal Transaction Cost.",
      warning: true,
    },
    reset: {
      label: "Manual reset",
      confirmation: "RESET",
      description: "Manual reset first flattens exposure, archives this Paper Account, and creates a new $10,000 Flat Start only after flattening succeeds.",
      danger: true,
    },
  };

  function openControlConfirmation(action) {
    const config = controls[action];
    if (!config) return;
    const { account, risk, live } = liveParts();
    const exposure = first(risk.gross_exposure, account.gross_exposure);
    const estimates = first(live.control_estimates, live.estimates, {});
    const estimate = first(estimates[action], action === "flatten" ? estimates.flatten_and_pause : undefined, action === "reset" ? estimates.manual_reset : undefined, {});
    const estimatedCost = first(estimate.estimated_cost, estimate.cost, live.estimated_flatten_cost);
    const exposureNotional = first(estimate.current_exposure, estimate.exposure_notional);
    const controlFacts = [
      element("div", { className: "control-fact" }, [element("span", { text: "Review current exposure" }), element("strong", { text: formatPercent(exposure) })]),
      exposureNotional !== undefined
        ? element("div", { className: "control-fact" }, [element("span", { text: "Exposure notional" }), element("strong", { text: formatMoney(exposureNotional) })])
        : null,
      element("div", { className: "control-fact" }, [element("span", { text: "Estimated Transaction Cost" }), element("strong", { text: formatMoney(estimatedCost) })]),
      element("div", { className: "control-fact" }, [element("span", { text: "Current lifecycle" }), element("strong", { text: first(account.state, account.lifecycle_state, live.state, "—") })]),
      element("div", { className: "control-fact" }, [element("span", { text: "Optimistic version" }), element("strong", { text: first(account.version, live.version, "—") })]),
    ].filter(Boolean);
    $("control-dialog-title").textContent = `Confirm ${config.label}`;
    $("control-confirmation").replaceChildren(
      element("p", { text: config.description }),
      element("div", { className: "control-facts" }, controlFacts),
      element("p", { className: "detail-intro", text: "This second step submits an idempotent Operator Intervention to the local account owner." }),
    );
    const confirm = $("confirm-control");
    confirm.textContent = `Confirm ${config.label}`;
    confirm.dataset.action = action;
    confirm.className = `button ${config.danger ? "button-danger" : config.warning ? "button-warning" : "button-secondary"}`;
    confirm.disabled = false;
    const dialog = $("control-dialog");
    if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
    else dialog.setAttribute("open", "");
  }

  function idempotencyKey() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `dashboard-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  async function submitControl(action) {
    const config = controls[action];
    const { live, account } = liveParts();
    if (!config) return;
    const confirm = $("confirm-control");
    confirm.disabled = true;
    confirm.textContent = "Submitting…";
    const requestKey = idempotencyKey();
    const body = {
      expected_version: first(account.version, live.version),
      confirmation: first(config.confirmation, config.label),
    };
    try {
      const csrfToken = state.csrfToken ?? await loadCsrf();
      const headers = {
        "Content-Type": "application/json",
        "X-Idempotency-Key": requestKey,
        "Idempotency-Key": requestKey,
        "X-Requested-With": "netgrowth-paper-dashboard",
        "X-CSRF-Token": csrfToken,
      };
      const result = await fetchJson(API.control(action), { method: "POST", headers, body: JSON.stringify(body) });
      $("control-dialog").close();
      showToast(first(result.message, result.detail, result.accepted ? "Control accepted" : `${config.label} submitted`));
      await refreshAll();
    } catch (error) {
      confirm.disabled = false;
      confirm.textContent = `Confirm ${config.label}`;
      showToast(`${config.label} was not accepted: ${error.message}`);
    }
  }

  function setView(view, options = {}) {
    if (!['live', 'history', 'system'].includes(view)) return;
    state.activeView = view;
    document.querySelectorAll("[role='tab']").forEach((tab) => {
      const selected = tab.dataset.view === view;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    document.querySelectorAll("[role='tabpanel']").forEach((panel) => {
      panel.hidden = panel.id !== `view-${view}`;
    });
    if (!options.fromHash && globalThis.location.hash !== `#${view}`) history.replaceState(null, "", `#${view}`);
    if (options.focus) $(`tab-${view}`).focus();
  }

  function requestedView() {
    const hashView = globalThis.location.hash.slice(1);
    if (['live', 'history', 'system'].includes(hashView)) return hashView;
    const pathView = globalThis.location.pathname.split('/').filter(Boolean).at(-1);
    return ['live', 'history', 'system'].includes(pathView) ? pathView : 'live';
  }

  function updateClocks() {
    const now = Date.now();
    $("kyiv-clock").textContent = `${formatDateTime(now, { seconds: true })} Kyiv`;
    $("utc-clock").textContent = `${formatDateTime(now, { utc: true, seconds: true })} UTC`;
    if (state.live) {
      const { freshness } = liveParts();
      const observedAt = first(freshness.observed_at, freshness.last_observation_at, state.live.as_of);
      const freshnessStatus = first(freshness.status, freshness.state, "Unknown");
      if (observedAt) $("freshness-pill").textContent = `${freshnessStatus} · ${duration((now - timestamp(observedAt)) / 1000)}`;
    }
  }

  function bindEvents() {
    document.querySelectorAll("[role='tab']").forEach((tab) => {
      tab.addEventListener("click", () => setView(tab.dataset.view));
      tab.addEventListener("keydown", (event) => {
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const views = ['live', 'history', 'system'];
        const direction = event.key === 'ArrowRight' ? 1 : -1;
        const index = (views.indexOf(state.activeView) + direction + views.length) % views.length;
        setView(views[index], { focus: true });
      });
    });
    globalThis.addEventListener("hashchange", () => setView(globalThis.location.hash.slice(1) || "live", { fromHash: true }));
    $("ticker-select").addEventListener("change", async (event) => {
      state.selectedTicker = event.target.value;
      try {
        await loadChart();
      } catch (error) {
        setConnectionError(error);
      }
    });
    document.querySelectorAll("[data-range]").forEach((button) => {
      button.addEventListener("click", async () => {
        state.selectedRange = button.dataset.range;
        document.querySelectorAll("[data-range]").forEach((rangeButton) => rangeButton.setAttribute("aria-pressed", String(rangeButton === button)));
        try {
          await loadChart();
        } catch (error) {
          setConnectionError(error);
        }
      });
    });
    $("account-select").addEventListener("change", async (event) => {
      state.selectedAccount = event.target.value;
      try {
        await loadHistory(state.selectedAccount);
      } catch (error) {
        setConnectionError(error);
      }
    });
    $("event-type-select").addEventListener("change", (event) => {
      state.eventType = event.target.value;
      renderHistory();
    });
    document.querySelectorAll("[data-control]").forEach((button) => button.addEventListener("click", () => openControlConfirmation(button.dataset.control)));
    $("confirm-control").addEventListener("click", () => submitControl($("confirm-control").dataset.action));
    document.querySelector("[data-close-event]").addEventListener("click", () => $("event-dialog").close());
    $("event-dialog").addEventListener("click", (event) => {
      if (event.target === $("event-dialog")) $("event-dialog").close();
    });
  }

  async function initialize() {
    state.chartRenderer = new globalThis.FinancialChart($("financial-chart"), {
      onMarker: (eventId) => openEvent(eventId),
    });
    bindEvents();
    setView(requestedView(), { fromHash: true });
    updateClocks();
    try {
      await loadCsrf();
    } catch (error) {
      showToast(`Lifecycle controls are unavailable: ${error.message}`);
    }
    await refreshAll();
    setInterval(updateClocks, 1000);
    setInterval(refreshAll, 60_000);
  }

  initialize();
})();
