(function registerFinancialChart(global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    grid: "#1b312c",
    text: "#91a9a1",
    faint: "#526c64",
    up: "#63e6be",
    down: "#ff786b",
    current: "#58c7dc",
    target: "#b29af8",
    equity: "#63e6be",
    cash: "#91a9a1",
    benchmark: "#b29af8",
    hold: "#58c7dc",
    drawdown: "#ff786b",
    gross: "#f4c95d",
    net: "#58c7dc",
    crosshair: "#d9e8e2",
    gap: "#365c52",
  };

  function svgElement(name, attributes = {}) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value));
    }
    return node;
  }

  function timestamp(value) {
    if (value instanceof Date) return value.getTime();
    if (typeof value === "number") return value < 1e12 ? value * 1000 : value;
    const parsed = Date.parse(String(value ?? ""));
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function number(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function formatNumber(value, maximumFractionDigits = 2) {
    if (!Number.isFinite(Number(value))) return "—";
    return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(Number(value));
  }

  function formatMoney(value) {
    if (!Number.isFinite(Number(value))) return "—";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(Number(value));
  }

  function formatPercent(value) {
    if (!Number.isFinite(Number(value))) return "—";
    return `${(Number(value) * 100).toFixed(2)}%`;
  }

  function formatTime(value, options = {}) {
    const date = new Date(timestamp(value));
    if (!Number.isFinite(date.getTime())) return "Unknown time";
    return new Intl.DateTimeFormat("en-GB", {
      timeZone: options.utc ? "UTC" : "Europe/Kiev",
      month: options.short ? undefined : "short",
      day: options.short ? undefined : "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: options.seconds ? "2-digit" : undefined,
      hourCycle: "h23",
      timeZoneName: options.zone ? "short" : undefined,
    }).format(date);
  }

  function extent(values, paddingRatio = 0.08) {
    const finite = values.map(Number).filter(Number.isFinite);
    if (!finite.length) return [0, 1];
    let minimum = Math.min(...finite);
    let maximum = Math.max(...finite);
    if (minimum === maximum) {
      const padding = Math.max(Math.abs(minimum) * paddingRatio, 1e-4);
      minimum -= padding;
      maximum += padding;
    } else {
      const padding = (maximum - minimum) * paddingRatio;
      minimum -= padding;
      maximum += padding;
    }
    return [minimum, maximum];
  }

  function linearScale(domainStart, domainEnd, rangeStart, rangeEnd) {
    const span = domainEnd - domainStart || 1;
    return (value) => rangeStart + ((number(value) - domainStart) / span) * (rangeEnd - rangeStart);
  }

  function pointTime(point) {
    return timestamp(point.time ?? point.timestamp ?? point.at ?? point.open_time);
  }

  function sortedPoints(points) {
    return Array.isArray(points) ? [...points].filter((point) => pointTime(point)).sort((a, b) => pointTime(a) - pointTime(b)) : [];
  }

  function appendText(parent, text, x, y, attributes = {}) {
    const node = svgElement("text", { x, y, fill: COLORS.text, "font-size": 10, ...attributes });
    node.textContent = text;
    parent.append(node);
    return node;
  }

  function pathFor(points, xScale, yScale, valueKey) {
    return points
      .filter((point) => Number.isFinite(Number(point[valueKey])))
      .map((point, index) => `${index ? "L" : "M"}${xScale(pointTime(point)).toFixed(2)},${yScale(point[valueKey]).toFixed(2)}`)
      .join(" ");
  }

  function markerColor(type) {
    const normalized = String(type).toLowerCase();
    if (normalized.includes("miss")) return COLORS.down;
    if (normalized.includes("fund")) return COLORS.gross;
    if (normalized.includes("intervention") || normalized.includes("reset") || normalized.includes("pause")) return COLORS.current;
    if (normalized.includes("fill") || normalized.includes("execution")) return COLORS.up;
    return COLORS.target;
  }

  class FinancialChart {
    constructor(container, options = {}) {
      this.container = container;
      this.onMarker = options.onMarker ?? (() => {});
      this.data = {};
      this.resizeObserver = "ResizeObserver" in global ? new ResizeObserver(() => this.render()) : null;
      this.resizeObserver?.observe(container);
    }

    update(data) {
      this.data = data && typeof data === "object" ? data : {};
      this.render();
    }

    destroy() {
      this.resizeObserver?.disconnect();
      this.container.replaceChildren();
    }

    render() {
      const candles = sortedPoints(this.data.candles ?? this.data.price ?? []);
      const weights = sortedPoints(this.data.weights ?? this.data.weight_points ?? []);
      const portfolio = sortedPoints(this.data.portfolio ?? this.data.account ?? this.data.panels ?? []);
      const equityPortfolio = portfolio.map((point) => ({
        ...point,
        hold_benchmark_equity: point.hold_benchmark?.equity ?? point.hold_benchmark_equity,
      }));
      const markers = sortedPoints(this.data.markers ?? this.data.events ?? []);
      const gaps = Array.isArray(this.data.gaps) ? this.data.gaps : [];
      const allTimes = [
        ...candles.map(pointTime),
        ...weights.map(pointTime),
        ...portfolio.map(pointTime),
        ...markers.map(pointTime),
      ].filter(Boolean);

      this.container.replaceChildren();
      this.container.setAttribute("aria-busy", "false");
      if (!allTimes.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No retained chart observations are available for this ticker and range.";
        this.container.append(empty);
        return;
      }

      const width = Math.max(this.container.clientWidth || 0, 760);
      const height = 690;
      const left = 62;
      const right = width - 18;
      const panels = {
        price: { top: 25, bottom: 270, label: `${this.data.ticker ?? "Selected ticker"} · price` },
        weights: { top: 292, bottom: 372, label: "Current / target weight" },
        equity: { top: 394, bottom: 492, label: "Equity / benchmarks" },
        drawdown: { top: 514, bottom: 575, label: "Drawdown" },
        exposure: { top: 597, bottom: 658, label: "Gross / net exposure" },
      };
      const timeExtent = extent(allTimes, 0);
      const xScale = linearScale(timeExtent[0], timeExtent[1], left, right);

      const scroll = document.createElement("div");
      scroll.className = "financial-chart-scroll";
      const svg = svgElement("svg", {
        viewBox: `0 0 ${width} ${height}`,
        width,
        height,
        role: "img",
        "aria-label": `${this.data.ticker ?? "Selected ticker"} synchronized paper account financial chart`,
      });
      const tooltip = document.createElement("div");
      tooltip.className = "chart-tooltip";
      tooltip.setAttribute("role", "note");
      tooltip.textContent = "Move across the chart for Kyiv and UTC-aligned values.";
      this.container.append(scroll, tooltip);
      scroll.append(svg);

      this.drawPanelFrames(svg, panels, left, right);
      this.drawGaps(svg, gaps, xScale, panels.price.top, panels.exposure.bottom);

      const priceValues = candles.flatMap((point) => [point.low, point.high, point.open, point.close]);
      const priceExtent = extent(priceValues);
      const priceScale = linearScale(priceExtent[0], priceExtent[1], panels.price.bottom, panels.price.top);
      this.drawCandles(svg, candles, xScale, priceScale, left, right);

      const weightValues = weights.flatMap((point) => [point.current ?? point.current_weight, point.target ?? point.target_weight]);
      const weightExtent = extent([...weightValues, 0]);
      const weightScale = linearScale(weightExtent[0], weightExtent[1], panels.weights.bottom, panels.weights.top);
      this.drawLine(svg, weights, xScale, weightScale, "current", "current_weight", COLORS.current);
      this.drawLine(svg, weights, xScale, weightScale, "target", "target_weight", COLORS.target, "4 3");

      const equityValues = portfolio.flatMap((point) => [
        point.equity ?? point.marked_equity,
        point.cash_benchmark,
        point.equal_weight_benchmark ?? point.passive_benchmark,
        point.hold_benchmark?.equity ?? point.hold_benchmark_equity,
      ]);
      const equityExtent = extent(equityValues);
      const equityScale = linearScale(equityExtent[0], equityExtent[1], panels.equity.bottom, panels.equity.top);
      this.drawLine(svg, equityPortfolio, xScale, equityScale, "equity", "marked_equity", COLORS.equity);
      this.drawLine(svg, equityPortfolio, xScale, equityScale, "cash_benchmark", null, COLORS.cash, "3 3");
      this.drawLine(svg, equityPortfolio, xScale, equityScale, "equal_weight_benchmark", "passive_benchmark", COLORS.benchmark);
      this.drawLine(svg, equityPortfolio, xScale, equityScale, "hold_benchmark_equity", null, COLORS.hold, "6 3");

      const drawdownExtent = extent([0, ...portfolio.map((point) => point.drawdown ?? point.current_drawdown)]);
      const drawdownScale = linearScale(drawdownExtent[0], drawdownExtent[1], panels.drawdown.top, panels.drawdown.bottom);
      this.drawLine(svg, portfolio, xScale, drawdownScale, "drawdown", "current_drawdown", COLORS.drawdown);

      const exposureExtent = extent([
        0,
        ...portfolio.flatMap((point) => [point.gross_exposure, point.net_exposure]),
      ]);
      const exposureScale = linearScale(exposureExtent[0], exposureExtent[1], panels.exposure.bottom, panels.exposure.top);
      this.drawLine(svg, portfolio, xScale, exposureScale, "gross_exposure", null, COLORS.gross);
      this.drawLine(svg, portfolio, xScale, exposureScale, "net_exposure", null, COLORS.net, "4 3");

      this.drawAxisLabels(svg, {
        left,
        right,
        timeExtent,
        panels,
        priceExtent,
        weightExtent,
        equityExtent,
        drawdownExtent,
        exposureExtent,
      });
      this.drawMarkers(svg, markers, candles, xScale, priceScale, panels.price);
      this.addCrosshair(svg, tooltip, {
        xScale,
        timeExtent,
        candles,
        weights,
        portfolio,
        left,
        right,
        top: panels.price.top,
        bottom: panels.exposure.bottom,
      });
    }

    drawPanelFrames(svg, panels, left, right) {
      for (const panel of Object.values(panels)) {
        svg.append(
          svgElement("rect", {
            x: left,
            y: panel.top,
            width: right - left,
            height: panel.bottom - panel.top,
            fill: "#091411",
            stroke: COLORS.grid,
          }),
        );
        appendText(svg, panel.label, 9, panel.top + 12, { "font-size": 9, "font-weight": 600 });
        for (let index = 1; index < 4; index += 1) {
          const y = panel.top + ((panel.bottom - panel.top) * index) / 4;
          svg.append(svgElement("line", { x1: left, x2: right, y1: y, y2: y, stroke: COLORS.grid, "stroke-width": 0.65 }));
        }
      }
      for (let index = 1; index < 6; index += 1) {
        const x = left + ((right - left) * index) / 6;
        svg.append(svgElement("line", { x1: x, x2: x, y1: panels.price.top, y2: panels.exposure.bottom, stroke: COLORS.grid, "stroke-width": 0.55 }));
      }
    }

    drawGaps(svg, gaps, xScale, top, bottom) {
      for (const gap of gaps) {
        const start = timestamp(gap.start ?? gap.started_at ?? gap.from);
        const end = timestamp(gap.end ?? gap.ended_at ?? gap.to);
        if (!start || !end) continue;
        const x = xScale(start);
        svg.append(
          svgElement("rect", {
            x,
            y: top,
            width: Math.max(1, xScale(end) - x),
            height: bottom - top,
            fill: COLORS.gap,
            opacity: 0.16,
            "aria-label": "Downtime or reconstructed observation gap",
          }),
        );
      }
    }

    drawCandles(svg, candles, xScale, yScale, left, right) {
      const candleWidth = Math.max(1.5, Math.min(11, ((right - left) / Math.max(candles.length, 1)) * 0.66));
      for (const candle of candles) {
        const x = xScale(pointTime(candle));
        const open = number(candle.open);
        const close = number(candle.close);
        const high = number(candle.high, Math.max(open, close));
        const low = number(candle.low, Math.min(open, close));
        const color = close >= open ? COLORS.up : COLORS.down;
        svg.append(svgElement("line", { x1: x, x2: x, y1: yScale(high), y2: yScale(low), stroke: color, "stroke-width": 1 }));
        svg.append(
          svgElement("rect", {
            x: x - candleWidth / 2,
            y: Math.min(yScale(open), yScale(close)),
            width: candleWidth,
            height: Math.max(1, Math.abs(yScale(open) - yScale(close))),
            fill: close >= open ? "#143c32" : "#462622",
            stroke: color,
            "stroke-width": 0.8,
          }),
        );
      }
    }

    drawLine(svg, points, xScale, yScale, primaryKey, aliasKey, color, dash = null) {
      const normalized = points.map((point) => ({ ...point, __value: point[primaryKey] ?? (aliasKey ? point[aliasKey] : undefined) }));
      const path = pathFor(normalized, xScale, yScale, "__value");
      if (!path) return;
      const attributes = { d: path, fill: "none", stroke: color, "stroke-width": 1.45, "vector-effect": "non-scaling-stroke" };
      if (dash) attributes["stroke-dasharray"] = dash;
      svg.append(svgElement("path", attributes));
    }

    drawAxisLabels(svg, options) {
      const formatExtent = (value, kind) => {
        if (kind === "money") return formatMoney(value);
        if (kind === "percent") return formatPercent(value);
        return formatNumber(value);
      };
      const labels = [
        [options.panels.price, options.priceExtent, "number"],
        [options.panels.weights, options.weightExtent, "percent"],
        [options.panels.equity, options.equityExtent, "money"],
        [options.panels.drawdown, options.drawdownExtent, "percent"],
        [options.panels.exposure, options.exposureExtent, "percent"],
      ];
      for (const [panel, values, kind] of labels) {
        appendText(svg, formatExtent(values[1], kind), options.left - 5, panel.top + 4, { "text-anchor": "end", "font-size": 8 });
        appendText(svg, formatExtent(values[0], kind), options.left - 5, panel.bottom, { "text-anchor": "end", "font-size": 8 });
      }
      const times = [options.timeExtent[0], (options.timeExtent[0] + options.timeExtent[1]) / 2, options.timeExtent[1]];
      const positions = [options.left, (options.left + options.right) / 2, options.right];
      const anchors = ["start", "middle", "end"];
      times.forEach((time, index) => {
        appendText(svg, `${formatTime(time, { short: true })} Kyiv`, positions[index], 679, { "text-anchor": anchors[index], "font-size": 8 });
      });
    }

    drawMarkers(svg, markers, candles, xScale, priceScale, pricePanel) {
      for (const marker of markers) {
        const time = pointTime(marker);
        const closest = this.nearestPoint(candles, time);
        const price = number(marker.price ?? marker.value ?? closest?.close, NaN);
        const x = xScale(time);
        const y = Number.isFinite(price) ? priceScale(price) : pricePanel.top + 25;
        const type = marker.type ?? marker.event_type ?? "Event";
        const label = marker.label ?? marker.title ?? marker.summary ?? "Material account event";
        const group = svgElement("g", {
          transform: `translate(${x},${y})`,
          role: "button",
          tabindex: 0,
          "aria-label": `${type} marker: ${label} at ${formatTime(time)} Kyiv`,
          cursor: "pointer",
        });
        const color = markerColor(type);
        group.append(
          svgElement("circle", {
            cx: 0,
            cy: 0,
            r: marker.material === false ? 5 : 7,
            fill: "#091411",
            stroke: color,
            "stroke-width": marker.material === false ? 1.3 : 2,
          }),
          svgElement("circle", { cx: 0, cy: 0, r: 2.2, fill: color }),
        );
        const activate = () => this.onMarker(marker.id ?? marker.event_id, marker);
        group.addEventListener("click", activate);
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            activate();
          }
        });
        svg.append(group);
      }
    }

    addCrosshair(svg, tooltip, options) {
      const line = svgElement("line", {
        x1: options.left,
        x2: options.left,
        y1: options.top,
        y2: options.bottom,
        stroke: COLORS.crosshair,
        "stroke-width": 0.8,
        opacity: 0,
        "pointer-events": "none",
      });
      svg.append(line);
      const update = (event) => {
        const bounds = svg.getBoundingClientRect();
        const rawX = ((event.clientX - bounds.left) / bounds.width) * Number(svg.getAttribute("width"));
        const x = Math.max(options.left, Math.min(options.right, rawX));
        const ratio = (x - options.left) / (options.right - options.left || 1);
        const time = options.timeExtent[0] + ratio * (options.timeExtent[1] - options.timeExtent[0]);
        const candle = this.nearestPoint(options.candles, time);
        const weight = this.nearestPoint(options.weights, time);
        const account = this.nearestPoint(options.portfolio, time);
        line.setAttribute("x1", x);
        line.setAttribute("x2", x);
        line.setAttribute("opacity", "0.7");
        tooltip.textContent = [
          `${formatTime(time, { seconds: true, zone: true })} · ${formatTime(time, { utc: true, seconds: true, zone: true })}`,
          `Price ${formatNumber(candle?.close)} · current ${formatPercent(weight?.current ?? weight?.current_weight)} · target ${formatPercent(weight?.target ?? weight?.target_weight)}`,
          `Equity ${formatMoney(account?.equity ?? account?.marked_equity)} · hold benchmark ${formatMoney(account?.hold_benchmark?.equity ?? account?.hold_benchmark_equity)} · cash benchmark ${formatMoney(account?.cash_benchmark)} · equal-weight ${formatMoney(account?.equal_weight_benchmark ?? account?.passive_benchmark)}`,
          `Drawdown ${formatPercent(account?.drawdown ?? account?.current_drawdown)} · gross ${formatPercent(account?.gross_exposure)} · net ${formatPercent(account?.net_exposure)}`,
        ].join("\n");
        tooltip.style.whiteSpace = "pre-line";
      };
      svg.addEventListener("pointermove", update);
      svg.addEventListener("pointerleave", () => line.setAttribute("opacity", "0"));
    }

    nearestPoint(points, time) {
      if (!points.length) return null;
      let closest = points[0];
      let distance = Math.abs(pointTime(closest) - time);
      for (let index = 1; index < points.length; index += 1) {
        const candidateDistance = Math.abs(pointTime(points[index]) - time);
        if (candidateDistance < distance) {
          closest = points[index];
          distance = candidateDistance;
        }
      }
      return closest;
    }
  }

  global.FinancialChart = FinancialChart;
})(window);
