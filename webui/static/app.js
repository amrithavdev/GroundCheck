"use strict";

const $ = (id) => document.getElementById(id);
// All text goes through textContent / text nodes, never innerHTML, so model
// output and SMS text can never inject markup.
const make = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const DEBIT = "Your account was debited ₹1,250. Your available balance is ₹8,400.";
const SAMPLES = [
  { label: "Correct claims", sms: DEBIT, tx: "1250", bal: "8400" },
  { label: "Swapped roles (the Gemma failure)", sms: DEBIT, tx: "8400", bal: "1250" },
  { label: "Amount not in message", sms: DEBIT, tx: "1300", bal: "8400" },
  { label: "Unclear wording", sms: "Your balance was debited ₹1,250.", tx: "1250", bal: "" },
];
const VERDICTS = {
  SUPPORTED: ["✓", "Supported", "ok"],
  UNSUPPORTED: ["✕", "Unsupported", "bad"],
  AMBIGUOUS: ["?", "Ambiguous", "warn"],
};
const FIELD_LABEL = { transaction_amount: "Transaction amount", available_balance: "Available balance" };
const MODES = {
  gemma: {
    note: "Sends the SMS to gemma3:1b through local Ollama. The model's answer is then checked against the SMS, never trusted.",
    button: "Extract & verify",
  },
  manual: {
    note: "No model involved. You type the claims; the verifier checks them against the SMS. Works without Ollama.",
    button: "Verify claims",
  },
};
let mode = "gemma";
let controller = null;

function setStatus(text, kind) {
  const box = $("status");
  box.replaceChildren();
  box.className = kind === "err" ? "status err" : "status";
  if (kind === "busy") box.append(make("span", "spin"));
  box.append(text);
}

function setMode(next) {
  mode = next;
  $("m-gemma").setAttribute("aria-pressed", String(next === "gemma"));
  $("m-manual").setAttribute("aria-pressed", String(next === "manual"));
  $("claims").hidden = next === "gemma";
  $("mode-note").textContent = MODES[next].note;
  $("run").textContent = MODES[next].button;
  setStatus("");
}

function busy(on) {
  $("run").disabled = on;
  $("cancel").hidden = !(on && mode === "gemma");
}

function sourceBlock(sms, snippets) {
  const ranges = [];
  for (const s of snippets) {
    const i = s ? sms.indexOf(s) : -1;
    if (i >= 0) ranges.push([i, i + s.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([r[0], r[1]]);
  }
  const box = make("p", "source");
  let pos = 0;
  for (const [a, b] of merged) {
    box.append(sms.slice(pos, a), make("mark", "", sms.slice(a, b)));
    pos = b;
  }
  box.append(sms.slice(pos));
  return box;
}

function badge(icon, label, tone) {
  const node = make("span", `badge ${tone}`);
  const sym = make("span", "", icon);
  sym.setAttribute("aria-hidden", "true");
  node.append(sym, label);
  return node;
}

function fieldCard(f) {
  const [icon, label, tone] = f.verdict
    ? VERDICTS[f.verdict]
    : f.status === "not_extracted" ? ["–", "Not checked", "none"] : ["!", "Invalid value", "none"];
  const card = make("article", "card");
  const head = make("div", "c-head");
  head.append(make("span", "c-title", FIELD_LABEL[f.field] || f.field), badge(icon, label, tone));
  const claim = make("p", "claim");
  claim.append(`Claimed as the ${f.role} amount: `, make("code", "", f.claimed ?? "nothing"));
  card.append(head, claim, make("p", "reason", f.reason));
  if (f.evidence.length) {
    for (const e of f.evidence) card.append(make("p", "quote", e));
  } else if (f.status === "checked") {
    card.append(make("p", "quote-none", "No matching evidence in the source."));
  }
  return card;
}

function render(data, sms) {
  const out = $("results");
  out.replaceChildren();
  const bar = make("div", "modebar");
  bar.append(
    make("span", "pill", data.mode === "gemma" ? `Live model output · ${data.model}` : "Manual claims · no model used"),
    make("span", "pill plain", "Checked by the rule-based verifier"),
  );
  out.append(bar, sourceBlock(sms, data.fields.flatMap((f) => f.evidence)));
  for (const f of data.fields) out.append(fieldCard(f));
  if (data.mode === "gemma") {
    const details = make("details");
    details.append(make("summary", "", "Raw model output (untrusted, shown as plain text)"), make("pre", "raw", data.raw));
    out.append(details);
  }
}

async function run() {
  const sms = $("sms").value.trim();
  if (!sms) return setStatus("Enter a bank SMS first.", "err");
  const request = mode === "gemma"
    ? ["/api/analyze", { sms }]
    : ["/api/verify", { sms, claims: { transaction_amount: $("c-tx").value, available_balance: $("c-bal").value } }];
  controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 130000);
  busy(true);
  setStatus(mode === "gemma"
    ? "Asking gemma3:1b through Ollama… the first request can be slow while the model loads."
    : "Verifying…", "busy");
  try {
    const res = await fetch(request[0], {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request[1]),
      signal: controller.signal,
    });
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-JSON error body */ }
    if (!res.ok) throw new Error(data.error || `Server returned HTTP ${res.status}.`);
    render(data, sms);
    setStatus("");
  } catch (err) {
    if (err.name === "AbortError") setStatus(timedOut ? "Timed out after 130 seconds." : "Cancelled.", "err");
    else if (err instanceof TypeError) setStatus("Could not reach the GroundCheck server. Is it still running?", "err");
    else setStatus(err.message, "err");
  } finally {
    clearTimeout(timer);
    busy(false);
    controller = null;
  }
}

function verdictBadge(value) {
  const [icon, label, tone] = VERDICTS[value] || ["!", value, "none"];
  return badge(icon, label, tone);
}

function renderBenchmark(box, data) {
  const s = data.summary;
  const pct = ((100 * s.passed) / s.total).toFixed(1);
  const tiles = make("div", "tiles");
  for (const [value, label, tone] of [
    [s.total, "Cases", ""],
    [`${s.passed}/${s.total}`, `Exact match (${pct}%)`, s.passed === s.total ? "ok" : "warn"],
    [s.false_acceptances, "False acceptances", s.false_acceptances === 0 ? "ok" : "bad"],
    [s.over_cautious, "Over-cautious", ""],
  ]) {
    const tile = make("div", "tile");
    tile.append(make("b", tone, String(value)), make("span", "", label));
    tiles.append(tile);
  }

  const dist = make("div", "dist");
  const legend = make("p", "legend");
  const counts = [["SUPPORTED", "ok"], ["UNSUPPORTED", "bad"], ["AMBIGUOUS", "warn"]];
  dist.setAttribute("role", "img");
  dist.setAttribute("aria-label", "Actual verdicts: " + counts.map(([k]) => `${k.toLowerCase()} ${s.actual[k] || 0}`).join(", "));
  for (const [key, tone] of counts) {
    const n = s.actual[key] || 0;
    const seg = make("i", tone);
    seg.style.setProperty("flex", String(n));
    dist.append(seg);
    legend.append(make("span", tone, `${VERDICTS[key][1]} ${n}`));
  }

  const select = make("select");
  select.id = "cat";
  select.append(make("option", "", "All categories"));
  select.firstChild.value = "";
  for (const c of [...new Set(data.cases.map((x) => x.category))]) {
    const opt = make("option", "", c);
    opt.value = c;
    select.append(opt);
  }
  const label = make("label", "", "Category");
  label.htmlFor = "cat";
  const tools = make("div", "tools");
  tools.append(label, select);

  const table = make("table");
  table.setAttribute("aria-label", "Benchmark cases");
  const headRow = make("tr");
  for (const h of ["Case", "Claim", "Expected", "Actual", "Result"]) headRow.append(make("th", "", h));
  const thead = make("thead");
  thead.append(headRow);
  const body = make("tbody");
  table.append(thead, body);

  const fill = () => {
    body.replaceChildren();
    for (const c of data.cases.filter((x) => !select.value || x.category === select.value)) {
      const row = make("tr");
      const name = make("td");
      name.append(make("span", "nm", c.name), make("small", "", c.source));
      const claim = make("td", "", `${c.amount} as ${c.role}`);
      const actual = make("td");
      actual.append(verdictBadge(c.actual), make("small", "", c.reason));
      const result = make("td");
      result.append(c.passed ? badge("✓", "Pass", "ok") : badge("✕", "Fail", "bad"));
      const expected = make("td");
      expected.append(verdictBadge(c.expected));
      row.append(name, claim, expected, actual, result);
      body.append(row);
    }
  };
  select.addEventListener("change", fill);
  fill();

  const scroll = make("div", "scroll");
  scroll.append(table);
  box.replaceChildren(tiles, dist, legend, tools, scroll);
}

async function loadBenchmark() {
  const box = $("b-body");
  try {
    const res = await fetch("/api/benchmark");
    if (!res.ok) throw new Error("bad status");
    renderBenchmark(box, await res.json());
  } catch (e) {
    box.replaceChildren(make("p", "empty", "Could not load benchmark results. Is the server running?"));
  }
}

async function loadEvaluation() {
  const box = $("e-body");
  if (!box) return;

  try {
    const response = await fetch("/api/evaluation");
    if (!response.ok) throw new Error("Evaluation unavailable");

    const data = await response.json();
    box.replaceChildren();

    const items = [
      ["Model", data.model],
      ["Baseline accuracy", `${(100 * data.baseline.accuracy.value).toFixed(1)}%`],
      ["Accepted precision", `${(100 * data.groundcheck.accepted_precision.value).toFixed(1)}%`],
      ["Coverage", `${(100 * data.groundcheck.coverage.value).toFixed(1)}%`],
      ["False acceptances", `${data.groundcheck.false_acceptances.num}/${data.groundcheck.false_acceptances.den}`],
      ["Correction success", `${(100 * data.correction.success.value).toFixed(1)}%`],
    ];

    const tiles = make("div", "tiles");
    for (const [label, value] of items) {
      const tile = make("div", "tile");
      tile.append(make("b", "", String(value)), make("span", "", label));
      tiles.append(tile);
    }

    box.append(tiles, make("p", "note", data.label));
  } catch {
    box.replaceChildren(make("p", "empty", "Could not load evaluation results."));
  }
}


function init() {
  $("sms").value = DEBIT;
  const chips = $("samples");
  for (const sample of SAMPLES) {
    const chip = make("button", "", sample.label);
    chip.type = "button";
    chip.addEventListener("click", () => {
      $("sms").value = sample.sms;
      $("c-tx").value = sample.tx;
      $("c-bal").value = sample.bal;
      setStatus("");
    });
    chips.append(chip);
  }
  $("m-gemma").addEventListener("click", () => setMode("gemma"));
  $("m-manual").addEventListener("click", () => setMode("manual"));
  $("run").addEventListener("click", run);
  $("cancel").addEventListener("click", () => controller && controller.abort());
  setMode("gemma");
  loadBenchmark();
  loadEvaluation();

}


init();
