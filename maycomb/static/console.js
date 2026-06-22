/* Maycomb operator console */
"use strict";

/* ---------------------------------------------------------------- helpers */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function pretty(v) {
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}
function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return `${Math.floor(diff)}초 전`;
  if (diff < 3600) return `${Math.floor(diff / 60)}분 전`;
  return d.toLocaleTimeString("ko-KR", { hour12: false });
}
function b62(n) {
  const A = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
  let out = "";
  const buf = new Uint8Array(n);
  crypto.getRandomValues(buf);
  for (const b of buf) out += A[b % 62];
  return out;
}
const newCallId = () => "call_" + b62(24);

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* noop */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ------------------------------------------------------------------ state */
const S = {
  ws: null,
  settings: {},
  filter: "pending",
  exchanges: new Map(),   // exchange_id -> summary row
  selected: null,
  detail: null,
  draftTimer: null,
  draftDirty: false,
  suppressAutosave: false,
  live: null,             // {sentReasoning, sentContent, pendTimer, locked}
};

/* --------------------------------------------------------------------- ws */
function wsConnect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/ws`);
  S.ws = ws;
  ws.onopen = () => $("#conn-dot").classList.add("on");
  ws.onclose = () => {
    $("#conn-dot").classList.remove("on");
    setTimeout(wsConnect, 1500);
  };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "hello") {
      S.settings = msg.settings || S.settings;
      renderTopbar(msg.tokenizer_active);
    } else if (msg.type === "exchange_new" || msg.type === "exchange_update") {
      if (msg.summary) onSummary(msg.summary, msg.type === "exchange_new");
    } else if (msg.type === "config") {
      refreshState();
    } else if (msg.type === "live_error") {
      toast(`LIVE 오류: ${msg.message}`, "err");
    } else if (msg.type === "live_ack" && msg.for === "live_finish") {
      liveStop("전송 마감됨");
    }
  };
}
function wsSend(obj) {
  if (!S.ws || S.ws.readyState !== 1) { toast("WebSocket이 연결되어 있지 않습니다", "err"); return false; }
  S.ws.send(JSON.stringify(obj));
  return true;
}

/* ------------------------------------------------------------------ queue */
function onSummary(sum, isNew) {
  S.exchanges.set(sum.exchange_id, sum);
  renderList();
  if (isNew && sum.status === "pending") {
    toast(`새 요청: ${sum.model ?? "?"} ${sum.stream ? "(stream)" : ""}`);
  }
  if (S.selected === sum.exchange_id) {
    renderHeader(sum);
    const terminal = ["completed", "aborted", "rejected", "injected"].includes(sum.status);
    const shown = S.detail?.summary?.status;
    if (terminal && shown !== sum.status) select(sum.exchange_id); // reload full detail
  }
}

async function refreshList() {
  const q = S.filter ? `?status=${S.filter}&limit=300` : "?limit=300";
  const data = await api(`/api/exchanges${q}`);
  if (S.filter) {
    // keep non-matching cached entries out of the way; rebuild from scratch
    S.exchanges.clear();
  }
  for (const row of data.exchanges) S.exchanges.set(row.exchange_id, row);
  renderList();
}

function renderList() {
  const rows = [...S.exchanges.values()]
    .filter((r) => !S.filter || r.status === S.filter)
    .sort((a, b) => {
      const pa = a.status === "pending" ? 0 : 1;
      const pb = b.status === "pending" ? 0 : 1;
      if (pa !== pb) return pa - pb;
      return pa === 0
        ? a.created_at.localeCompare(b.created_at)        // pending: FIFO
        : b.created_at.localeCompare(a.created_at);       // done: recent first
    });
  const pending = [...S.exchanges.values()].filter((r) => r.status === "pending").length;
  $("#pending-badge").textContent = `대기 ${pending}`;
  document.title = pending ? `(${pending}) Maycomb 콘솔` : "Maycomb 콘솔";

  $("#xlist").innerHTML = rows.map((r) => `
    <li data-id="${r.exchange_id}" class="${r.exchange_id === S.selected ? "sel" : ""}">
      <div class="xi-top">
        <span class="st ${r.status}"></span>
        <b>${esc(r.model ?? "?")}</b>
        ${r.stream ? '<span class="chip">stream</span>' : ""}
        ${r.partial ? '<span class="chip warn">partial</span>' : ""}
        <span class="spacer"></span>
        <span class="hint">${fmtTime(r.created_at)}</span>
      </div>
      <div class="xi-prev">${esc(r.preview ?? "")}</div>
    </li>`).join("");
  $$("#xlist li").forEach((li) => li.onclick = () => select(li.dataset.id));
}

/* ----------------------------------------------------------------- detail */
async function select(xid) {
  try {
    const detail = await api(`/api/exchanges/${xid}`);
    S.selected = xid;
    S.detail = detail;
    S.draftDirty = false;
    if (S.live) liveStop(null);
    renderDetail();
    renderList();
  } catch (e) {
    toast(`불러오기 실패: ${e.message}`, "err");
  }
}

function renderTopbar(tokenizerActive) {
  $("#readonly-badge").hidden = !S.settings.read_only;
  if (tokenizerActive) $("#tokenizer-badge").textContent = `tokenizer: ${tokenizerActive}`;
}

function renderHeader(sum) {
  $("#x-status").textContent = sum.status;
  $("#x-status").className = `chip ${sum.status}`;
  $("#x-id").textContent = sum.chatcmpl_id;
  $("#x-model").textContent = sum.model ?? "?";
  const rtm = sum.runtime;
  let prog = "";
  if (rtm) {
    if (rtm.chunks_sent) prog = `chunk ${rtm.chunks_sent} 전송됨`;
    if (rtm.injection && !rtm.injection.applied) {
      prog += `  [주입 대기: ${rtm.injection.kind}]`;
    }
  }
  $("#x-progress").textContent = prog;

  const f = sum.flags || {};
  const badges = [];
  if (f.stream) badges.push(`<span class="chip">SSE</span>`);
  if (f.include_usage) badges.push(`<span class="chip">include_usage</span>`);
  if (f.response_format && f.response_format !== "text")
    badges.push(`<span class="chip warn">response_format: ${esc(f.response_format)}</span>`);
  if (f.tool_choice && f.tool_choice !== "auto")
    badges.push(`<span class="chip warn">tool_choice: ${esc(f.tool_choice)} (표시만, 강제 안 함)</span>`);
  if (f.n_tools) badges.push(`<span class="chip">tools ${f.n_tools}</span>`);
  if (f.max_tokens != null) badges.push(`<span class="chip">max_tokens ${esc(f.max_tokens)}</span>`);
  if (f.parallel_tool_calls != null)
    badges.push(`<span class="chip">parallel_tool_calls: ${f.parallel_tool_calls}</span>`);
  if (f.reasoning_request)
    badges.push(`<span class="chip purple">reasoning 요청: ${esc(JSON.stringify(f.reasoning_request))}</span>`);
  for (const u of f.unknown_fields || [])
    badges.push(`<span class="chip bad">비표준 필드: ${esc(u)}</span>`);
  for (const t of sum.tags || []) badges.push(`<span class="chip purple">${esc(t)}</span>`);
  $("#x-badges").innerHTML = badges.join(" ");
}

function renderDetail() {
  const { summary, request } = S.detail;
  $("#empty-state").hidden = true;
  $("#xview").hidden = false;
  renderHeader(summary);
  renderConversation(request);
  renderParams(request);
  renderTools(request);
  $("#tab-events").innerHTML = `<button id="btn-load-events" class="sm">이벤트 불러오기</button>`;
  $("#btn-load-events").onclick = loadEvents;
  loadRaw();
  renderResult();
  renderComposer();
}

function contentToHtml(content) {
  if (content == null) return `<span class="hint">null</span>`;
  if (typeof content === "string") return `<div class="pre">${esc(content)}</div>`;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (part && part.type === "text")
        return `<div class="pre">${esc(part.text)}</div>`;
      return `<details><summary>비텍스트 파트: ${esc(part?.type ?? "?")}</summary>
        <pre class="json">${esc(pretty(part))}</pre></details>`;
    }).join("");
  }
  return `<pre class="json">${esc(pretty(content))}</pre>`;
}

function renderConversation(request) {
  const messages = request?.messages || [];
  const html = messages.map((m, i) => {
    const role = m?.role ?? "?";
    let body = "";
    if (m?.reasoning_content)
      body += `<details class="reasoning-block"><summary>reasoning_content</summary>
        <div class="pre">${esc(m.reasoning_content)}</div></details>`;
    body += contentToHtml(m?.content);
    for (const tc of m?.tool_calls || []) {
      let args = tc?.function?.arguments ?? "";
      try { args = pretty(JSON.parse(args)); } catch { /* keep raw */ }
      body += `<div class="tcall"><span class="fn">${esc(tc?.function?.name)}</span>
        <span class="hint">${esc(tc?.id ?? "")}</span>
        <pre class="json">${esc(args)}</pre></div>`;
    }
    const meta = [];
    if (m?.name) meta.push(`name=${esc(m.name)}`);
    if (m?.tool_call_id) meta.push(`tool_call_id=<code>${esc(m.tool_call_id)}</code>`);
    const long = (typeof m?.content === "string" && m.content.length > 1500);
    const inner = `
      <div class="msg-head"><span class="role ${esc(role)}">${esc(role)}</span>
        <span class="hint">#${i}${meta.length ? " · " + meta.join(" ") : ""}</span></div>
      <div class="msg-body">${body}</div>`;
    return long
      ? `<div class="msg"><details><summary class="msg-head">
           <span class="role ${esc(role)}">${esc(role)}</span>
           <span class="hint">#${i} (${m.content.length}자 — 펼치기)</span></summary>
           <div class="msg-body">${body}</div></details></div>`
      : `<div class="msg">${inner}</div>`;
  }).join("");
  $("#tab-conv").innerHTML = html || `<span class="hint">메시지 없음</span>`;
}

function renderParams(request) {
  if (!request) { $("#tab-params").innerHTML = `<span class="hint">파싱 실패한 요청</span>`; return; }
  const unknown = new Set(S.detail.unknown_fields || []);
  const skip = new Set(["messages", "tools"]);
  const rows = Object.keys(request).filter((k) => !skip.has(k)).sort().map((k) => {
    const cls = unknown.has(k) ? "unknown-row" : "";
    const v = typeof request[k] === "object" && request[k] !== null
      ? `<pre class="json">${esc(pretty(request[k]))}</pre>`
      : `<code>${esc(JSON.stringify(request[k]))}</code>`;
    const mark = unknown.has(k) ? " ⚠ 비표준" : "";
    return `<tr class="${cls}"><td>${esc(k)}${mark}</td><td>${v}</td></tr>`;
  }).join("");
  $("#tab-params").innerHTML =
    `<table class="kv">${rows}</table>
     <p class="hint">temperature/top_p/seed 등은 표시만 하며 동작에 영향을 주지 않습니다 (§2.1).</p>`;
}

function renderTools(request) {
  const tools = request?.tools || [];
  $("#tools-count").textContent = tools.length ? `(${tools.length})` : "";
  $("#tools-datalist").innerHTML = tools
    .map((t) => `<option value="${esc(t?.function?.name ?? "")}">`).join("");
  $("#tab-tools").innerHTML = tools.length
    ? tools.map((t) => `
      <div class="tool-def">
        <b class="mono">${esc(t?.function?.name ?? "?")}</b>
        <div class="hint">${esc(t?.function?.description ?? "")}</div>
        <details><summary>parameters</summary>
          <pre class="json">${esc(pretty(t?.function?.parameters))}</pre></details>
      </div>`).join("")
    : `<span class="hint">요청에 tools 없음</span>`;
}

async function loadEvents() {
  const data = await api(`/api/exchanges/${S.selected}/events`);
  const rows = data.events.map((e) => `
    <div class="ev-row">
      <span class="hint">#${e.seq}</span>
      <span class="ev-type">${esc(e.type)}</span>
      <span class="hint">${esc(e.ts)}</span>
      <details><summary>data</summary><pre class="json">${esc(pretty(e.data))}</pre></details>
    </div>`).join("");
  $("#tab-events").innerHTML = rows || `<span class="hint">이벤트 없음</span>`;
}

async function loadRaw() {
  try {
    const res = await fetch(`/api/exchanges/${S.selected}/raw`);
    $("#raw-pre").textContent = await res.text();
  } catch { $("#raw-pre").textContent = "(raw 없음)"; }
}

function renderResult() {
  const card = $("#result-card");
  const ev = S.detail.result;
  if (!ev) { card.hidden = true; card.innerHTML = ""; return; }
  card.hidden = false;
  card.className = "";
  let html = "";
  if (ev.type === "response_submitted") {
    const d = ev.data, msg = d.response?.choices?.[0]?.message || {};
    html += `<h4>전송된 응답 ${d.meta?.partial ? '<span class="chip warn">partial</span>' : ""}
      <span class="chip">${esc(d.meta?.mode ?? "")}</span></h4>`;
    if (msg.reasoning_content)
      html += `<details class="reasoning-block" open><summary>reasoning_content</summary>
        <div class="pre">${esc(msg.reasoning_content)}</div></details>`;
    if (msg.content) html += `<div class="pre">${esc(msg.content)}</div>`;
    for (const tc of msg.tool_calls || []) {
      html += `<div class="tcall"><span class="fn">${esc(tc.function?.name)}</span>
        <span class="hint">${esc(tc.id)}</span>
        <pre class="json">${esc(tc.function?.arguments ?? "")}</pre></div>`;
    }
    html += `<div class="hint">finish_reason=${esc(d.response?.choices?.[0]?.finish_reason)}
      · usage: ${esc(JSON.stringify(d.usage))}</div>`;
    if (d.meta?.operator_note) html += `<div class="hint">노트: ${esc(d.meta.operator_note)}</div>`;
  } else if (ev.type === "exchange_aborted") {
    card.className = "aborted";
    html += `<h4>중단됨 — ${esc(ev.data?.reason)}</h4>`;
    if (ev.data?.reconstruction)
      html += `<details><summary>전송분 재구성</summary>
        <pre class="json">${esc(pretty(ev.data.reconstruction))}</pre></details>`;
  } else if (ev.type === "request_rejected") {
    card.className = "aborted";
    html += `<h4>요청 거부 (HTTP ${esc(ev.data?.status)})</h4>
      <pre class="json">${esc(pretty(ev.data?.error))}</pre>`;
  }
  card.innerHTML = html;
}

/* --------------------------------------------------------------- composer */
function renderComposer() {
  const sum = S.detail.summary;
  const comp = $("#composer");
  const editable = sum.status === "pending";
  const activeStream = sum.status === "active" && sum.stream;
  comp.hidden = !(editable || activeStream);
  if (comp.hidden) return;

  $("#stream-controls").style.display = sum.stream ? "" : "none";
  $("#btn-live").hidden = !sum.stream;
  $("#btn-submit").disabled = !editable;
  $("#btn-live").disabled = !editable;
  $("#btn-save").disabled = !editable && !S.live;
  $("#btn-abort-cancel").style.display = sum.stream ? "none" : "";
  $("#btn-abort-stop").style.display = sum.stream ? "" : "none";
  $("#btn-abort-length").style.display = sum.stream ? "" : "none";
  $("#btn-abort-hard").style.display = sum.stream ? "" : "none";
  $("#comp-warnings").hidden = true;

  S.suppressAutosave = true;
  $("#c-rate").value = S.settings.tokens_per_second ?? 30;
  $("#c-ttft").value = S.settings.ttft_ms ?? 0;
  $("#c-interleave").checked = !!S.settings.stress_interleave;

  const draft = S.detail.draft;
  $("#c-reasoning").value = draft?.reasoning_content ?? "";
  $("#c-content").value = draft?.content ?? "";
  $("#c-finish").value = draft?.finish_reason ?? "auto";
  $("#c-tags").value = "";
  $("#c-note").value = "";
  $("#c-bypass").checked = false;
  $("#draft-rev").textContent = draft?.revision ? `rev ${draft.revision}` : "";
  $("#c-toolcalls").innerHTML = "";
  for (const tc of draft?.tool_calls || []) {
    const args = "arguments_raw" in tc ? tc.arguments_raw
      : tc.arguments_obj !== undefined ? pretty(tc.arguments_obj) : "";
    addToolCallCard(tc.name, args, tc.id);
  }
  $("#c-reasoning").readOnly = false;
  $("#c-content").readOnly = false;
  $("#live-panel").hidden = true;
  S.suppressAutosave = false;
}

function addToolCallCard(name = "", args = "{}", id = null) {
  const card = document.createElement("div");
  card.className = "tc-card";
  card.dataset.callId = id || newCallId();
  card.innerHTML = `
    <div class="tc-head">
      <input class="tc-name" placeholder="function 이름" list="tools-datalist" value="${esc(name)}">
      <code class="tc-id">${esc(card.dataset.callId)}</code>
      <button class="sm tc-live" hidden>LIVE 전송</button>
      <button class="sm tc-del">✕</button>
    </div>
    <textarea class="tc-args mono" rows="3" spellcheck="false">${esc(args)}</textarea>
    <div class="tc-valid hint"></div>`;
  $("#c-toolcalls").appendChild(card);
  const argsEl = card.querySelector(".tc-args");
  const validEl = card.querySelector(".tc-valid");
  const check = () => {
    try { JSON.parse(argsEl.value); validEl.textContent = "JSON OK"; validEl.className = "tc-valid hint ok"; }
    catch (e) { validEl.textContent = `JSON 오류: ${e.message}`; validEl.className = "tc-valid hint bad"; }
    markDirty();
  };
  argsEl.addEventListener("input", check);
  card.querySelector(".tc-name").addEventListener("input", markDirty);
  card.querySelector(".tc-del").onclick = () => { card.remove(); markDirty(); };
  card.querySelector(".tc-live").onclick = () => {
    wsSend({ type: "live_tool_call", exchange_id: S.selected,
             id: card.dataset.callId,
             name: card.querySelector(".tc-name").value,
             arguments: argsEl.value });
    lockForToolStage();
  };
  card.querySelector(".tc-live").hidden = !S.live;
  check();
}

function collectDraft() {
  return {
    reasoning_content: $("#c-reasoning").value,
    content: $("#c-content").value,
    finish_reason: $("#c-finish").value,
    tool_calls: $$(".tc-card").map((card) => ({
      id: card.dataset.callId,
      name: card.querySelector(".tc-name").value.trim(),
      arguments: card.querySelector(".tc-args").value,
    })),
  };
}

function markDirty() {
  if (S.suppressAutosave) return;
  S.draftDirty = true;
  clearTimeout(S.draftTimer);
  S.draftTimer = setTimeout(saveDraft, 1200);
}

async function saveDraft(silent = true) {
  if (!S.selected) return;
  const status = S.detail?.summary?.status;
  if (status !== "pending" && !S.live) return;
  clearTimeout(S.draftTimer);
  try {
    const r = await api(`/api/exchanges/${S.selected}/draft`, {
      method: "POST",
      body: { draft: collectDraft(), mode: $("#c-mode").value },
    });
    S.draftDirty = false;
    $("#draft-rev").textContent = `rev ${r.revision} 저장됨`;
    if (!silent) toast(`드래프트 rev ${r.revision} 저장`);
  } catch (e) {
    if (!silent) toast(`드래프트 저장 실패: ${e.message}`, "err");
  }
}

/* ------------------------------------------------------------------ submit */
async function submit() {
  if (!S.selected) return;
  clearTimeout(S.draftTimer);
  const draft = collectDraft();
  const bypass = $("#c-bypass").checked;
  try {
    const v = await api(`/api/exchanges/${S.selected}/validate`, {
      method: "POST", body: { draft } });
    const box = $("#comp-warnings");
    const lines = [];
    if (v.blockers.length) lines.push("⛔ " + v.blockers.join("\n⛔ "));
    if (v.warnings.length) lines.push("⚠ " + v.warnings.join("\n⚠ "));
    box.hidden = lines.length === 0;
    box.textContent = lines.join("\n");
    if (v.blockers.length && !bypass) {
      toast("검증 실패 — 차단 항목을 고치거나 '강제 우회'를 체크하세요", "err");
      return;
    }
  } catch (e) {
    toast(`검증 호출 실패: ${e.message}`, "err");
    return;
  }
  const sum = S.detail.summary;
  const body = {
    draft,
    mode: sum.stream ? $("#c-mode").value : "json",
    pacing: {
      tokens_per_second: parseFloat($("#c-rate").value) || undefined,
      ttft_ms: parseInt($("#c-ttft").value, 10) || 0,
      interleave: $("#c-interleave").checked,
    },
    meta: {
      operator_note: $("#c-note").value,
      tags: $("#c-tags").value.split(",").map((s) => s.trim()).filter(Boolean),
      validation_bypass: bypass,
    },
  };
  try {
    const r = await api(`/api/exchanges/${S.selected}/submit`, { method: "POST", body });
    toast(`제출됨 (finish_reason=${r.finish_reason})`);
    for (const w of r.warnings || []) toast(w, "warn");
  } catch (e) {
    toast(`제출 실패: ${e.message}`, "err");
  }
}

/* ------------------------------------------------------------ abort/inject */
async function abort(kind, finishReason) {
  try {
    await api(`/api/exchanges/${S.selected}/abort`, {
      method: "POST", body: { kind, finish_reason: finishReason } });
    toast(kind === "hard" ? "강제 절단됨" : "마감 처리됨");
  } catch (e) { toast(`중단 실패: ${e.message}`, "err"); }
}

async function inject(kind, params = {}) {
  try {
    await api(`/api/exchanges/${S.selected}/inject`, {
      method: "POST", body: { kind, params } });
    toast(`주입됨: ${kind}`);
  } catch (e) { toast(`주입 실패: ${e.message}`, "err"); }
}

/* -------------------------------------------------------------- live mode */
function liveStart() {
  if (!S.selected) return;
  if (!wsSend({ type: "live_start", exchange_id: S.selected })) return;
  // live relays *new* typing; existing draft text is the baseline and is not sent
  S.live = {
    sentReasoning: $("#c-reasoning").value,
    sentContent: $("#c-content").value,
    locked: false,
    pendTimer: null,
  };
  if (S.live.sentReasoning || S.live.sentContent) {
    toast("LIVE: 기존 드래프트 텍스트는 전송되지 않습니다. 새로 타이핑한 내용만 중계됩니다", "warn");
  }
  $("#live-panel").hidden = false;
  $("#live-stage").textContent = "stage: role 전송됨";
  $("#btn-submit").disabled = true;
  $("#btn-live").disabled = true;
  $$(".tc-live").forEach((b) => b.hidden = false);
}

function liveStop(msg) {
  S.live = null;
  $("#live-panel").hidden = true;
  $$(".tc-live").forEach((b) => b.hidden = true);
  if (msg) toast(msg);
}

function liveFlush(field) {
  if (!S.live) return;
  const isReasoning = field === "reasoning_content";
  const el = isReasoning ? $("#c-reasoning") : $("#c-content");
  const key = isReasoning ? "sentReasoning" : "sentContent";
  const delta = el.value.slice(S.live[key].length);
  if (!delta) return;
  if (wsSend({ type: "live_text", exchange_id: S.selected, field, text: delta })) {
    S.live[key] = el.value;
    if (!isReasoning && !S.live.locked) {
      S.live.locked = true;
      $("#c-reasoning").readOnly = true;  // §4.2: content 시작 후 reasoning 잠금
      $("#live-stage").textContent = "stage: content";
    } else if (isReasoning) {
      $("#live-stage").textContent = "stage: reasoning";
    }
  }
}

function lockForToolStage() {
  if (!S.live) return;
  $("#c-reasoning").readOnly = true;
  $("#c-content").readOnly = true;
  $("#live-stage").textContent = "stage: tool_calls";
}

function liveInput(field) {
  if (!S.live) { markDirty(); return; }
  const isReasoning = field === "reasoning_content";
  const el = isReasoning ? $("#c-reasoning") : $("#c-content");
  const sent = S.live[isReasoning ? "sentReasoning" : "sentContent"];
  if (!el.value.startsWith(sent)) {
    el.value = sent;  // append-only: revert edits into already-sent text
    toast("LIVE 모드: 이미 전송된 텍스트는 수정할 수 없습니다", "warn");
    return;
  }
  clearTimeout(S.live.pendTimer);
  S.live.pendTimer = setTimeout(() => liveFlush(field), 350);
}

/* ---------------------------------------------------------------- settings */
async function refreshState() {
  const st = await api("/api/state");
  S.settings = st.settings;
  $("#version").textContent = `v${st.version}`;
  renderTopbar(st.tokenizer_active);
}

function openSettings() {
  const s = S.settings;
  $("#s-readonly").checked = !!s.read_only;
  $("#s-alias").checked = !!s.reasoning_alias;
  $("#s-stress").checked = !!s.stress_interleave;
  $("#s-tokenizer").value = s.tokenizer ?? "o200k_base";
  $("#s-rate").value = s.tokens_per_second ?? 30;
  $("#s-ttft").value = s.ttft_ms ?? 0;
  $("#s-keepalive").value = s.keepalive_interval ?? 15;
  $("#s-auth").value = s.auth_mode ?? "any";
  $("#s-apikey").value = "";
  $("#s-cancel").value = s.nonstream_cancel ?? "error500";
  $("#settings-dialog").showModal();
}

async function saveSettings() {
  const body = {
    read_only: $("#s-readonly").checked,
    reasoning_alias: $("#s-alias").checked,
    stress_interleave: $("#s-stress").checked,
    tokenizer: $("#s-tokenizer").value,
    tokens_per_second: parseFloat($("#s-rate").value) || 30,
    ttft_ms: parseInt($("#s-ttft").value, 10) || 0,
    keepalive_interval: parseFloat($("#s-keepalive").value) || 0,
    auth_mode: $("#s-auth").value,
    nonstream_cancel: $("#s-cancel").value,
  };
  if ($("#s-apikey").value) body.api_key = $("#s-apikey").value;
  try {
    const r = await api("/api/config", { method: "PUT", body });
    S.settings = r.settings;
    renderTopbar(r.tokenizer_active);
    toast("설정 저장됨");
  } catch (e) { toast(`설정 저장 실패: ${e.message}`, "err"); }
}

/* ------------------------------------------------------------------- init */
function bind() {
  $$("#queue-tabs button").forEach((b) => b.onclick = () => {
    $$("#queue-tabs button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    S.filter = b.dataset.f;
    refreshList().catch((e) => toast(e.message, "err"));
  });
  $$("#tabs button").forEach((b) => b.onclick = () => {
    $$("#tabs button").forEach((x) => x.classList.remove("on"));
    $$(".tab").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    $(`#tab-${b.dataset.tab}`).classList.add("on");
  });
  $("#btn-refresh").onclick = () => refreshList().catch((e) => toast(e.message, "err"));
  $("#btn-settings").onclick = openSettings;
  $("#btn-save-settings").onclick = saveSettings;
  $("#btn-add-tc").onclick = () => { addToolCallCard(); markDirty(); };
  $("#btn-save").onclick = () => saveDraft(false);
  $("#btn-submit").onclick = submit;
  $("#btn-live").onclick = liveStart;
  $("#btn-live-finish-auto").onclick = () =>
    wsSend({ type: "live_finish", exchange_id: S.selected, finish_reason: "auto" });
  $("#btn-live-finish-stop").onclick = () =>
    wsSend({ type: "live_finish", exchange_id: S.selected, finish_reason: "stop" });
  $("#btn-abort-stop").onclick = () => abort("graceful", "stop");
  $("#btn-abort-length").onclick = () => abort("graceful", "length");
  $("#btn-abort-hard").onclick = () => {
    if (confirm("finish chunk 없이 TCP를 끊습니다. 진행할까요?")) abort("hard");
  };
  $("#btn-abort-cancel").onclick = () => {
    if (confirm("non-streaming 요청을 취소합니다 (설정된 방식: " +
        (S.settings.nonstream_cancel ?? "error500") + ")")) abort("cancel");
  };
  $$("#danger-zone [data-inject]").forEach((b) =>
    b.onclick = () => {
      if (b.dataset.inject === "rate_limit") {
        const ra = prompt("Retry-After (초)", "30");
        if (ra === null) return;
        inject("rate_limit", { retry_after: parseInt(ra, 10) || 30 });
      } else inject(b.dataset.inject);
    });
  $("#btn-inject-delay").onclick = () => {
    const ms = prompt("TTFT 지연(ms) — 제출 시 적용, keep-alive도 보류됩니다", "10000");
    if (ms === null) return;
    inject("delay", { delay_ms: parseInt(ms, 10) || 10000 });
  };
  $("#btn-inject-cut").onclick = () => {
    const n = prompt("몇 번째 chunk 후 절단할까요? (비우면 즉시 절단)", "");
    if (n === null) return;
    inject("stream_cut", n.trim() === "" ? {} : { after_chunks: parseInt(n, 10) || 1 });
  };
  $("#c-reasoning").addEventListener("input", () => liveInput("reasoning_content"));
  $("#c-content").addEventListener("input", () => liveInput("content"));
  ["#c-finish", "#c-mode"].forEach((sel) =>
    $(sel).addEventListener("change", markDirty));
}

async function init() {
  bind();
  try { await refreshState(); } catch (e) { toast(`서버 상태 조회 실패: ${e.message}`, "err"); }
  try { await refreshList(); } catch (e) { toast(e.message, "err"); }
  wsConnect();
}
init();
