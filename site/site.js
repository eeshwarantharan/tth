// Renders the project page from window.TTH (site/data.js), which is generated from committed results.
(function () {
  "use strict";
  var d = window.TTH;
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function $(id) { return document.getElementById(id); }

  // ---- Trace replay -------------------------------------------------------------------------
  var tabs = document.querySelector("#replay .term-tabs");
  var body = document.querySelector("#replay .term-body");
  var note = $("replay-note");
  var timers = [];
  var current = d.replays[0];

  function stepNode(s) {
    if (s.kind === "stage") {
      var box = el("div", "staged");
      box.appendChild(el("span", "cmd", s.command));
      box.appendChild(el("span", "badge", "[" + s.tier + "] staged on the prompt, not run"));
      return box;
    }
    if (s.kind === "end") return el("div", "none", "✕ " + s.text);
    var row = el("div", "step");
    if (s.kind === "empty") {
      row.appendChild(el("span", "tick bad", "✕"));
      row.appendChild(el("span", "in", s.text));
      return row;
    }
    row.appendChild(el("span", "tick", "✓"));
    var line = el("span", "");
    line.appendChild(el("span", "tool", s.tool + " "));
    line.appendChild(el("span", "in", s.input));
    row.appendChild(line);
    if (s.output) row.appendChild(el("span", "out", s.output));
    return row;
  }

  function verdictNode(r) {
    return el("div", "verdict", r.correct ? "verifier: correct" : "verifier: wrong");
  }

  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function show(r, animate) {
    clearTimers();
    current = r;
    body.textContent = "";
    Array.prototype.forEach.call(tabs.querySelectorAll("[role=tab]"), function (b) {
      b.setAttribute("aria-selected", String(b.dataset.id === r.id));
    });
    var nodes = [el("div", "req", r.request)].concat(r.steps.map(stepNode), [verdictNode(r)]);
    if (!animate || REDUCED) {
      nodes.forEach(function (n) { body.appendChild(n); });
    } else {
      nodes.forEach(function (n, i) {
        timers.push(setTimeout(function () { body.appendChild(n); }, i * 750));
      });
    }
    note.textContent = r.note;
    note.appendChild(el("span", "prov", "Recorded eval run · " + r.model + " · task " + r.task + " · episode " + r.episode));
  }

  d.replays.forEach(function (r) {
    var b = el("button", "", r.label);
    b.type = "button";
    b.setAttribute("role", "tab");
    b.dataset.id = r.id;
    b.addEventListener("click", function () { show(r, false); });
    tabs.appendChild(b);
  });
  var play = el("button", "play", "▶ Replay");
  play.type = "button";
  play.addEventListener("click", function () { show(current, true); });
  tabs.appendChild(play);
  show(current, false);

  // ---- Safety ---------------------------------------------------------------------------------
  var TIER = {
    "read-only": "● read-only",
    "changes-files": "◐ changes files",
    "needs-review": "▲ needs review",
    "denied": "✕ denied"
  };

  [[d.safety.red_team_cases, "red-team cases"], [d.safety.fuzz_commands, "adversarial commands"]].forEach(function (s) {
    var box = el("div");
    box.appendChild(el("b", "", s[0].toLocaleString("en-US")));
    box.appendChild(el("span", "", s[1]));
    $("safety-stats").appendChild(box);
  });

  function outcome(g) {
    var a = el("div", "action");
    if (g.staged === null) a.appendChild(el("b", "", "Refuses to stage it. "));
    else if (g.rewritten) {
      a.appendChild(el("b", "", "Rewrites to "));
      a.appendChild(el("code", "", g.staged));
      a.appendChild(document.createTextNode(". "));
    } else if (g.tier === "read-only") a.appendChild(el("b", "", "Stages it, and can test it first. "));
    else a.appendChild(el("b", "", "Stages it with this badge. "));
    if (g.reason) {
      // Command names stay lowercase, because that's how you'd type them.
      var startsWithCommand = g.reason.split(" ")[0] === g.command.split(/\s/)[0].toLowerCase();
      var text = startsWithCommand ? g.reason : g.reason.charAt(0).toUpperCase() + g.reason.slice(1);
      a.appendChild(document.createTextNode(text + "."));
    }
    return a;
  }

  function renderGallery(filter) {
    var g = $("gallery");
    g.textContent = "";
    d.gallery.filter(function (x) { return !filter || x.tier === filter; }).forEach(function (x) {
      var row = el("div", "gallery-row");
      row.appendChild(el("code", "", x.command));
      row.appendChild(el("span", "tier tier-" + x.tier, TIER[x.tier]));
      row.appendChild(outcome(x));
      g.appendChild(row);
    });
  }

  [["", "All"]].concat(Object.keys(TIER).map(function (k) { return [k, TIER[k]]; })).forEach(function (f, i) {
    var b = el("button", "", f[1]);
    b.type = "button";
    b.setAttribute("aria-pressed", String(i === 0));
    b.addEventListener("click", function () {
      Array.prototype.forEach.call($("filters").children, function (c) { c.setAttribute("aria-pressed", "false"); });
      b.setAttribute("aria-pressed", "true");
      renderGallery(f[0]);
    });
    $("filters").appendChild(b);
  });
  renderGallery("");

  // ---- Charts ---------------------------------------------------------------------------------
  var NS = "http://www.w3.org/2000/svg";
  var tip = $("tip");

  function svg(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function hover(target, text) {
    target.setAttribute("tabindex", "0");
    target.setAttribute("aria-label", text);
    function place(x, y) {
      tip.textContent = text;
      tip.hidden = false;
      var w = tip.offsetWidth;
      tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, x + 12)) + "px";
      tip.style.top = Math.max(8, y - tip.offsetHeight - 10) + "px";
    }
    target.addEventListener("pointermove", function (e) { place(e.clientX, e.clientY); });
    target.addEventListener("pointerleave", function () { tip.hidden = true; });
    target.addEventListener("focus", function () {
      var r = target.getBoundingClientRect();
      place(r.left, r.top);
    });
    target.addEventListener("blur", function () { tip.hidden = true; });
  }

  function sci(p) {
    if (p >= 0.01) return p.toFixed(2);
    var e = Math.floor(Math.log10(p));
    return (p / Math.pow(10, e)).toFixed(1) + "e" + e;
  }

  var SUPER = { 2: "⁻²", 4: "⁻⁴", 6: "⁻⁶", 8: "⁻⁸" };

  function renderTags(width) {
    var host = $("chart-tags");
    host.textContent = "";
    var narrow = width < 460;
    var left = narrow ? 8 : 124, right = 26, rowH = narrow ? 72 : 50, top = 6, bottom = 30;
    var h = top + rowH * d.tags.length + bottom;
    var s = svg("svg", { viewBox: "0 0 " + width + " " + h, role: "img", "aria-label": "Probability of the opening tag and of end of turn, per arm" });
    var lo = -9, hi = 0;
    function x(p) { return left + (Math.log10(p) - lo) / (hi - lo) * (width - left - right); }
    [-8, -6, -4, -2, 0].forEach(function (e) {
      var X = x(Math.pow(10, e));
      s.appendChild(svg("line", { x1: X, x2: X, y1: top, y2: h - bottom, stroke: "var(--line)", "stroke-width": 1 }));
      s.appendChild(svg("text", { x: X, y: h - 10, "text-anchor": "middle" }, e === 0 ? "1" : "10" + SUPER[-e]));
    });
    d.tags.forEach(function (t, i) {
      var cy = top + rowH * i + rowH * (narrow ? 0.66 : 0.5);
      if (narrow) s.appendChild(svg("text", { class: "row", x: left, y: cy - 24 }, t.arm));
      else s.appendChild(svg("text", { class: "row", x: 0, y: cy + 4 }, t.arm));
      var a = x(t.open_tag), b = x(t.end_turn);
      s.appendChild(svg("line", { x1: Math.min(a, b), x2: Math.max(a, b), y1: cy, y2: cy, stroke: "var(--line)", "stroke-width": 2 }));
      s.appendChild(svg("circle", { cx: a, cy: cy, r: 5, fill: "var(--series-a)", stroke: "var(--panel)", "stroke-width": 2 }));
      s.appendChild(svg("rect", { x: b - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: "var(--series-b)", stroke: "var(--panel)", "stroke-width": 2 }));
      s.appendChild(svg("text", { class: "value", x: a, y: cy + 19, "text-anchor": "middle" }, sci(t.open_tag)));
      s.appendChild(svg("text", { class: "value", x: Math.min(b, width - right + 8), y: cy - 10, "text-anchor": "middle" }, sci(t.end_turn)));
      var hitA = svg("circle", { cx: a, cy: cy, r: 14, fill: "transparent" });
      hover(hitA, t.arm + ": opening tag p = " + sci(t.open_tag));
      s.appendChild(hitA);
      var hitB = svg("circle", { cx: b, cy: cy, r: 14, fill: "transparent" });
      hover(hitB, t.arm + ": end of turn p = " + sci(t.end_turn));
      s.appendChild(hitB);
    });
    host.appendChild(s);
  }

  function armLabel(f) { return f.arm + (f.forced ? ", forced" : ""); }

  function renderForced(width) {
    var host = $("chart-forced");
    host.textContent = "";
    var narrow = width < 460;
    var left = narrow ? 8 : 168, right = 16, rowH = narrow ? 44 : 32, top = 6, bottom = 28;
    var h = top + rowH * d.forced.length + bottom;
    var s = svg("svg", { viewBox: "0 0 " + width + " " + h, role: "img", "aria-label": "Correct out of 20 tasks per arm, with 95 percent intervals" });
    function x(v) { return left + v * (width - left - right); }
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      s.appendChild(svg("line", { x1: x(v), x2: x(v), y1: top, y2: h - bottom, stroke: "var(--line)", "stroke-width": 1 }));
      s.appendChild(svg("text", { x: x(v), y: h - 8, "text-anchor": v === 1 ? "end" : v === 0 ? "start" : "middle" }, Math.round(v * 100) + "%"));
    });
    d.forced.forEach(function (f, i) {
      var cy = top + rowH * i + rowH * (narrow ? 0.72 : 0.5);
      if (narrow) s.appendChild(svg("text", { class: "row", x: left, y: cy - 12 }, armLabel(f)));
      else s.appendChild(svg("text", { class: "row", x: 0, y: cy + 4 }, armLabel(f)));
      var rate = f.correct / f.n;
      s.appendChild(svg("line", { x1: x(f.ci[0]), x2: x(f.ci[1]), y1: cy, y2: cy, stroke: "var(--series-a)", "stroke-width": 2, "stroke-linecap": "round" }));
      s.appendChild(svg("circle", { cx: x(rate), cy: cy, r: 5, fill: "var(--series-a)", stroke: "var(--panel)", "stroke-width": 2 }));
      s.appendChild(svg("text", { class: "value", x: x(f.ci[1]) + 8, y: cy + 4 }, f.correct + "/" + f.n));
      var hit = svg("rect", { x: x(f.ci[0]) - 6, y: cy - 12, width: x(f.ci[1]) - x(f.ci[0]) + 12, height: 24, fill: "transparent" });
      hover(hit, armLabel(f) + ": " + f.correct + "/" + f.n + ", 95% interval " + Math.round(f.ci[0] * 100) + "–" + Math.round(f.ci[1] * 100) + "%");
      s.appendChild(hit);
    });
    host.appendChild(s);
  }

  function renderCharts() {
    renderTags($("chart-tags").clientWidth || 480);
    renderForced($("chart-forced").clientWidth || 480);
  }
  renderCharts();
  var lastW = window.innerWidth;
  window.addEventListener("resize", function () {
    if (window.innerWidth !== lastW) { lastW = window.innerWidth; renderCharts(); }
  });

  // ---- Findings, in numbers from the data ------------------------------------------------------
  function arm(name, forced) {
    return d.forced.filter(function (f) { return f.arm === name && f.forced === forced; })[0];
  }
  var baseU = arm("Untrained base", false), baseF = arm("Untrained base", true);
  var shell = arm("Shell training only", true), chessShell = arm("Chess, then shell", true);
  var table = arm("Table, then shell", true), chess = arm("Chess only", true);
  $("finding-training").textContent = "The trained arms got " + chess.correct + " to " + shell.correct + " of 20. The untrained model got " +
    baseF.correct + " with the forced format and " + baseU.correct + " without it. All of these intervals overlap.";
  var staged = shell.staged + chessShell.staged + table.staged;
  var verified = shell.verified + chessShell.verified + table.verified;
  $("finding-behaviour").textContent = "The arms with shell training dry-run about " + shell.dry_runs.toFixed(1) + " to " + chessShell.dry_runs.toFixed(1) + " times per task, against " +
    baseU.dry_runs.toFixed(1) + " for base, and " + verified + " of the " + staged + " commands they staged were verified. But they ran out of attempts in " +
    shell.budget + " to " + chessShell.budget + " of 20 tasks without committing. Looping is now the top failure.";
  $("finding-chess").textContent = "Chess then shell (" + chessShell.correct + "/20) tied shell only (" + shell.correct + "/20) and table then shell (" + table.correct +
    "/20). Chess only verified the least, " + chess.verified + " of " + chess.staged + " staged. The sample is small, so chess stays as an optional gym environment and gets a proper multi-seed test later.";

  var head = ["Arm", "Recorded (100 tasks)", "This run (20 tasks)", "Malformed", "Out of attempts", "Staged / verified", "Dry-runs per task", "Seconds per task"];
  var recorded = {};
  d.recorded.forEach(function (r) { recorded[r.arm] = r.correct + "/" + r.n; });
  var t = $("forced-table");
  var thead = el("thead"), htr = el("tr");
  head.forEach(function (h, i) { htr.appendChild(el("th", i ? "num" : "", h)); });
  thead.appendChild(htr);
  t.appendChild(thead);
  var tb = el("tbody");
  d.forced.forEach(function (f) {
    var tr = el("tr");
    tr.appendChild(el("td", "", armLabel(f)));
    [f.forced && f.arm === "Untrained base" ? "not run" : recorded[f.arm], f.correct + "/" + f.n, f.malformed, f.budget, f.staged + " / " + f.verified, f.dry_runs.toFixed(1), f.seconds.toFixed(1)]
      .forEach(function (v) { tr.appendChild(el("td", "num", String(v))); });
    tb.appendChild(tr);
  });
  t.appendChild(tb);

  // ---- Roadmap ----------------------------------------------------------------------------------
  d.roadmap.forEach(function (p) {
    var state = /^(done|live)/i.test(p.status) ? "done" : /^next/i.test(p.status) ? "next" : "";
    var li = el("li", state);
    li.appendChild(el("span", "ph", p.phase));
    li.appendChild(el("span", "", p.scope));
    li.appendChild(el("span", "st", (state === "done" ? "✓ " : "") + p.status));
    $("road").appendChild(li);
  });
})();
