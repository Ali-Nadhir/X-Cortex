const $ = (s) => document.querySelector(s);

const state = {
  cfg: null,
  sessions: [],
  current: null,
  running: false,
  pickedFile: null,
  lastStudyEl: null,
};

let EMPTY_HTML = "";
let uidCounter = 0;
const uid = () => `g${++uidCounter}`;

const BAND_COLOR = { high: "#fff", indeterminate: "#ccc", low: "#aaa" };


document.addEventListener("DOMContentLoaded", init);

const DEFAULT_PREFS = { sort: "recent", group: "", pinnedFirst: false, hideCompleted: false, pins: [], compact: false, concise: false, notifications: true };
let prefs = { ...DEFAULT_PREFS };
try { prefs = { ...prefs, ...JSON.parse(localStorage.getItem("xcortex.ui") || "{}") }; } catch (_) {}
if (!Array.isArray(prefs.pins)) prefs.pins = [];
state.records = [];
state.error = false;
state.loading = false;
state.readingFiles = false;
state.lastRequest = null;
let activePanel = null;

async function init() {
  EMPTY_HTML = $("#emptyState").outerHTML;
  bind();
  applyPreferences();
  renderSessions();
  renderContext();
  await connect();
}

async function connect() {
  try {
    state.cfg = await getJSON("/api/config");
    $("#modeBadge").textContent = state.cfg.mode === "mock" ? "SIMULATION MODE" : "LIVE MODEL MODE";
    renderModels();
    setBusy(false);
    await refreshSessions();
  } catch (e) {
    state.cfg = null;
    $("#sysStatus").textContent = "SERVER UNAVAILABLE";
    $("#sysLed").className = "led alert";
    $("#modelInfo").innerHTML = '<button class="menu-row" id="reconnect"><span>Server unavailable</span><small>Retry connection</small></button>';
    $("#reconnect").addEventListener("click", connect);
    snackbar("SERVER UNAVAILABLE — check your X-Cortex server.");
  }
}

function closeDrawer(restoreFocus = false) {
  document.querySelectorAll(".popout").forEach(el => { el.hidden = true; });
  document.querySelectorAll("[data-panel]").forEach(el => el.setAttribute("aria-expanded", "false"));
  $("#scrim").classList.remove("show");
  if (restoreFocus && activePanel) document.querySelector('[data-panel="' + activePanel + '"]').focus();
  activePanel = null;
}

function openPanel(name) {
  const wasOpen = activePanel === name;
  closeDrawer();
  if (wasOpen) return;
  activePanel = name;
  $("#" + name + "Panel").hidden = false;
  document.querySelector('[data-panel="' + name + '"]').setAttribute("aria-expanded", "true");
  $("#scrim").classList.add("show");
  if (name === "files") renderContext();
  const focusable = $("#" + name + "Panel").querySelector("input, button, summary");
  if (focusable) focusable.focus();
}

function bind() {
  for (const id of ["#newAnalysis", "#homeBtn"]) $(id).addEventListener("click", () => {
    if (state.running || state.loading || state.readingFiles) return snackbar("Wait for the current study to finish.");
    closeDrawer(); resetView();
  });
  document.querySelectorAll("[data-panel]").forEach(el => el.addEventListener("click", () => openPanel(el.dataset.panel)));
  $("#attachBtn").addEventListener("click", () => $("#fileInput").click());
  $("#fileInput").addEventListener("change", onPick);
  $("#attachmentRemove").addEventListener("click", clearAttachment);
  $("#sendBtn").addEventListener("click", onSend);
  $("#historyInput").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); onSend(); }
  });
  $("#historyInput").addEventListener("input", e => { autosize(e); renderContext(); });
  $("#sessionSearch").addEventListener("input", renderSessions);
  $("#scrim").addEventListener("click", () => closeDrawer());
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(true); });
  document.querySelectorAll("[data-sort]").forEach(el => el.addEventListener("click", () => { prefs.sort = el.dataset.sort; savePreferences(); }));
  document.querySelectorAll("[data-group]").forEach(el => el.addEventListener("click", () => { prefs.group = prefs.group === el.dataset.group ? "" : el.dataset.group; savePreferences(); }));
  for (const [id, key] of [["pinnedFirst", "pinnedFirst"], ["hideCompleted", "hideCompleted"], ["densityToggle", "compact"], ["reportDetail", "concise"], ["notificationsToggle", "notifications"]]) {
    $("#" + id).addEventListener("click", () => { prefs[key] = !prefs[key]; savePreferences(); });
  }
  $("#resetArrangement").addEventListener("click", () => { Object.assign(prefs, { sort: "recent", group: "", pinnedFirst: false, hideCompleted: false }); savePreferences(); });
  $("#app").addEventListener("dragover", e => { if (e.dataTransfer.types.includes("Files")) e.preventDefault(); });
  $("#app").addEventListener("drop", e => { e.preventDefault(); if (!state.running) onPick({ target: { files: e.dataTransfer.files } }); });
}

function savePreferences() {
  try { localStorage.setItem("xcortex.ui", JSON.stringify(prefs)); } catch (_) {}
  applyPreferences(); renderSessions();
}
function applyPreferences() {
  document.body.classList.toggle("compact", prefs.compact);
  document.body.classList.toggle("concise", prefs.concise);
  document.querySelectorAll("[data-sort]").forEach(el => el.setAttribute("aria-pressed", String(el.dataset.sort === prefs.sort)));
  document.querySelectorAll("[data-group]").forEach(el => el.setAttribute("aria-pressed", String(el.dataset.group === prefs.group)));
  for (const [id, key] of [["pinnedFirst", "pinnedFirst"], ["hideCompleted", "hideCompleted"], ["densityToggle", "compact"], ["reportDetail", "concise"], ["notificationsToggle", "notifications"]]) $("#" + id).setAttribute("aria-pressed", String(prefs[key]));
  $("#densityToggle small").textContent = prefs.compact ? "Compact" : "Comfortable";
  $("#reportDetail small").textContent = prefs.concise ? "Impression and limitations" : "Detailed output";
  $("#notificationsToggle small").textContent = prefs.notifications ? "Completion alerts on" : "Completion alerts off";
}

function menuItem(title, detail, selected = false) {
  return '<div class="menu-row' + (selected ? ' selected' : '') + '"><span>' + esc(title) + '</span><small>' + esc(detail) + '</small></div>';
}
function renderModels() {
  const cfg = state.cfg;
  if (!cfg) return;
  $("#modelInfo").innerHTML = menuItem("Primary reasoning model", cfg.mode === "mock" ? "Simulated backend · no model inference" : cfg.models.reasoner === "medgemma" ? cfg.models.medgemma : cfg.models.reasoner, true);
  $("#toolsInfo").innerHTML = cfg.tools.map(t => menuItem(t.label, cfg.mode === "mock" ? "Simulated" : "Enabled", true)).join("");
}
function renderContext() {
  const current = state.current;
  let html = "";
  if (current) html += menuItem(current.title || "Radiograph", "Primary study", true);
  if (state.pickedFile) html += menuItem(state.pickedFile.name, "Ready to upload", true);
  html += state.records.map(r => menuItem(r.name, "JSON · pending clinical context")).join("");
  const history = $("#historyInput").value.trim() || (current && current.history);
  if (history) html += '<details class="menu-row"><summary>Clinical context</summary><small>History / records / question</small><p class="context-history">' + esc(history) + '</p></details>';
  if (current && current.preview_url) html += '<a href="' + esc(current.preview_url) + '" target="_blank" rel="noopener"><img class="context-preview" src="' + esc(current.preview_url) + '" alt="Current radiograph"/></a>';
  $("#contextFiles").innerHTML = html || '<p class="sessions-empty">No files attached yet.</p>';
  const count = Number(!!current) + Number(!!state.pickedFile) + state.records.length + Number(!!history);
  $("#contextSummary").innerHTML = count + ' ITEMS IN CONTEXT<small>One primary radiograph per study. JSON records are included as text in the clinical context.</small>';
}
async function onPick(e) {
  if (state.running || state.loading || state.readingFiles) return;
  const files = Array.from(e.target.files || []);
  const images = files.filter(f => /\.(png|jpe?g|tiff?|bmp|dcm|dicom|dc3)$/i.test(f.name));
  if (images.length > 1) return snackbar("Choose one primary radiograph per study. JSON records can be added alongside it.");
  state.readingFiles = true;
  $("#sendBtn").disabled = true;
  for (const file of files) {
    if (/\.json$/i.test(file.name)) {
      if (file.size > 1024 * 1024) { snackbar("JSON records must be smaller than 1 MB."); continue; }
      try {
        const text = JSON.stringify(JSON.parse(await file.text()), null, 2);
        state.records = state.records.filter(r => r.name !== file.name);
        state.records.push({ name: file.name, text });
      } catch (_) { snackbar("Invalid JSON record: " + file.name); }
    } else if (images.includes(file)) state.pickedFile = file;
    else snackbar("Unsupported file: " + file.name);
  }
  state.readingFiles = false;
  $("#sendBtn").disabled = false;
  updateAttachment();
}
function updateAttachment() {
  const names = [state.pickedFile && state.pickedFile.name, ...state.records.map(r => r.name)].filter(Boolean);
  $("#attachmentName").textContent = names.join(" · ");
  $("#attachmentChip").hidden = names.length === 0;
  renderContext();
}
function clearAttachment() {
  state.pickedFile = null; state.records = [];
  $("#fileInput").value = "";
  updateAttachment();
}
function autosize(e) {
  const el = e.target || $("#historyInput");
  el.style.height = "76px";
  el.style.height = Math.min(Math.max(el.scrollHeight, 76), 168) + "px";
}
function composedHistory() {
  return [$("#historyInput").value.trim(), ...state.records.map(r => "Record: " + r.name + "\n" + r.text)].filter(Boolean).join("\n\n");
}
async function onSend() {
  if (state.running || state.loading || state.readingFiles) return;
  if (!state.cfg) { await connect(); if (!state.cfg) return; }
  const text = composedHistory();
  if (state.pickedFile) return startAnalysis(state.pickedFile, text);
  if (state.current && !state.current.reported) return analyzeCurrent(text || state.current.history || "");
  if (state.current && text) return ask(text);
  snackbar("Attach a radiograph to begin.");
}
function showConversation() {
  hideEmpty();
  $("#app").classList.add("in-conversation");
  $(".chat-mascot").hidden = false;
  $("#screenKicker").textContent = "ACTIVE CASE / CONVERSATION";
  $("#screenTitle").textContent = "Study conversation";
  $("#screenDescription").textContent = "The case is active. X Cortex is reading the image together with the records and context already provided.";
  $("#historyInput").placeholder = "Continue the case, ask a follow-up, or add new context…";
  $("#sendBtn").textContent = "SEND";
}
function clearComposer() {
  clearAttachment(); $("#historyInput").value = ""; autosize({ target: $("#historyInput") });
}
async function startAnalysis(file, history) {
  clearSnackbar();
  state.error = false; state.running = true; setBusy(true); showConversation();
  state.lastRequest = { type: "upload", file, text: history };
  const form = new FormData(); form.append("file", file);
  let up;
  try { up = await getJSON("/api/upload", { method: "POST", body: form }); }
  catch (e) { const block = addAssistantBlock(true); markBlockError(block, "Upload failed: " + e.message); finish(); return; }
  state.current = { id: up.session_id, reported: false, title: file.name, preview_url: up.preview_url, history };
  addUserStudy(up, history);
  clearComposer();
  state.running = false;
  await analyzeCurrent(history);
}
async function analyzeCurrent(history) {
  if (state.running || !state.current) return;
  state.current.history = history;
  clearComposer();
  renderContext();
  state.lastRequest = { type: "analyze", text: history };
  await runRequest("/api/analyze", { session_id: state.current.id, history }, false);
}
async function ask(text) {
  state.lastRequest = { type: "ask", text };
  addUserText(text);
  clearComposer();
  await runRequest("/api/ask", { session_id: state.current.id, question: text }, true);
}
async function runRequest(url, payload, textMode) {
  clearSnackbar();
  state.error = false; state.running = true; setBusy(true); showConversation();
  const block = addAssistantBlock(textMode);
  try {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error("Request failed (" + res.status + ")");
    await streamSSE(res, block, state.current.id);
    if (!block.done && !state.error) throw new Error("The connection ended before the response completed.");
  } catch (e) { markBlockError(block, e.message); }
  finish();
  await refreshSessions();
}
function finish() {
  state.running = false; setBusy(false); renderContext();
  if (state.lastStudyEl) state.lastStudyEl.classList.remove("scanning");
}
function setBusy(busy) {
  for (const id of ["sendBtn", "attachBtn", "newAnalysis", "homeBtn", "attachmentRemove", "fileInput"]) $("#" + id).disabled = busy;
  $("#thread").setAttribute("aria-busy", String(busy));
  $("#sysLed").className = "led " + (busy ? "run" : state.error ? "alert" : "ok");
  $("#sysStatus").textContent = !state.cfg ? "SERVER UNAVAILABLE" : busy ? "ANALYZING CASE" : state.error ? "ATTENTION NEEDED" : state.current ? "CASE READY" : "READY FOR STUDY";
}
async function streamSSE(res, block, sessionId) {
  if (!res.body) throw new Error("Streaming is unavailable.");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const event = parseSSE(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 2);
        if (event) handleEvent(event, block, sessionId);
      }
      if (done) { const event = parseSSE(buffer); if (event) handleEvent(event, block, sessionId); break; }
    }
  } finally { reader.releaseLock(); }
}

function parseSSE(raw) {
  let name = "message";
  let dataText = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) dataText += line.slice(5).trim();
  }
  if (!dataText) return null;
  try {
    return { event: name, data: JSON.parse(dataText) };
  } catch (e) {
    return { event: name, data: { text: dataText } };
  }
}

function handleEvent(ev, block, sessionId) {
  const { event, data } = ev;
  if (event === "done") {
    block.done = true;
    const typing = block.stream.querySelector(".typing");
    if (typing) typing.remove();
  } else if (event === "start") {
    ensureToolCards(block);
  } else if (event === "stage") {
    updateToolCard(block, data);
  } else if (event === "evidence") {
    fillToolCard(block, data);
  } else if (event === "token") {
    appendToken(block, data.text || "");
  } else if (event === "report") {
    state.current.reported = true;
    if (state.lastStudyEl) state.lastStudyEl.classList.remove("scanning");
    renderReport(block, data, sessionId);
    if (prefs.notifications) snackbar("Study analysis complete.");
  } else if (event === "error") {
    markBlockError(block, data.message || "UNKNOWN FAULT");
  }
  scrollBottom();
}

function hideEmpty() {
  const el = $("#emptyState");
  if (el) el.remove();
}

function resetView() {
  if (state.running || state.loading || state.readingFiles) return;
  clearSnackbar();
  state.current = null;
  state.lastStudyEl = null;
  state.lastRequest = null;
  state.error = false;
  const thread = $("#thread");
  thread.innerHTML = EMPTY_HTML;
  $("#app").classList.remove("in-conversation");
  $(".chat-mascot").hidden = true;
  $("#screenKicker").textContent = "FIRST CONTACT / STUDY INTAKE";
  $("#screenTitle").textContent = "Bring the image. Bring the context.";
  $("#screenDescription").textContent = "The X-ray is the starting point. The surrounding clinical information is what makes the interpretation useful.";
  $("#historyInput").placeholder = "Add symptoms, records, reports, prior scans, and your question…";
  $("#sendBtn").textContent = "START ANALYSIS";
  clearComposer();
  renderContext();
  setBusy(false);
  markActiveSession(null);
  scrollBottom();
}

function addUserStudy(up, history) {
  const row = document.createElement("div");
  row.className = "msg user";
  const chips = Object.entries(up.meta || {})
    .filter(([k]) => k !== "pixel_spacing_mm")
    .map(([k, v]) => `<span class="meta-chip">${esc(k)}: ${esc(String(v))}</span>`)
    .join("");
  row.innerHTML = `
    <div class="msg-body">
      <div class="message-label">YOU</div>
      <div class="study-card">
        <div class="study-img"><img src="${esc(up.preview_url)}" alt="radiograph"/></div>
        <div class="study-info">
          ${history ? `<div class="study-history">${mdLite(history)}</div>` : `<div class="study-history" style="color:var(--faint)">NO CLINICAL HISTORY PROVIDED</div>`}
          ${chips ? `<div class="meta-chips">${chips}</div>` : ""}
        </div>
      </div>
    </div>`;
  $("#thread").appendChild(row);
  state.lastStudyEl = row.querySelector(".study-card");
  scrollBottom();
}

function addUserText(text) {
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="msg-body"><div class="message-label">YOU</div><div class="user-text">${mdLite(text)}</div></div>`;
  $("#thread").appendChild(row);
  scrollBottom();
}

function addAssistantBlock(textMode) {
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.innerHTML = `
    <div class="msg-body"><div class="message-label">X CORTEX <span class="bone" aria-hidden="true"></span></div></div>`;
  const body = row.querySelector(".msg-body");
  const tools = document.createElement("div");
  tools.className = "tools";
  const stream = document.createElement("div");
  stream.className = "stream";
  stream.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  body.appendChild(stream);
  if (!textMode) body.appendChild(tools);
  $("#thread").appendChild(row);
  scrollBottom();
  return { root: row, body, tools, stream };
}

function ensureToolCards(block) {
  if (!state.cfg || !block.tools || block.tools.children.length) return;
  for (const tool of state.cfg.tools) {
    const card = document.createElement("div");
    card.className = "tool-card pending";
    card.dataset.tool = tool.id;
    card.innerHTML = `
      <div class="tool-head">
        <span class="tool-name">${esc(tool.label.toUpperCase())}</span>
        <span class="tool-summary"></span>
        <span class="tool-state"><span class="led"></span><span class="ts-text">QUEUED</span></span>
      </div>
      <div class="m3-progress" hidden><i></i></div>
      <div class="tool-metrics"></div>`;
    block.tools.appendChild(card);
  }
}

function toolCard(block, toolId) {
  return block.tools ? block.tools.querySelector(`[data-tool="${toolId}"]`) : null;
}

function updateToolCard(block, data) {
  const card = toolCard(block, data.tool);
  if (!card) return;
  const led = card.querySelector(".led");
  const text = card.querySelector(".ts-text");
  const progress = card.querySelector(".m3-progress");
  if (data.status === "running") {
    card.className = "tool-card running";
    led.className = "led run";
    text.textContent = "RUNNING";
    progress.hidden = false;
  } else if (data.status === "done") {
    card.className = "tool-card complete";
    led.className = "led ok";
    text.textContent = "COMPLETE";
    progress.hidden = true;
    card.querySelector(".tool-summary").textContent = data.summary || "";
  }
}

function fillToolCard(block, data) {
  const card = toolCard(block, data.tool);
  if (!card) return;
  const metrics = card.querySelector(".tool-metrics");
  metrics.innerHTML = metricsHTML(data.payload, data.tool);
  metrics.classList.add("open");
}

function metricsHTML(payload, toolId) {
  if (!payload) return "";
  if (toolId === "triage") {
    const cls = "b-" + (payload.band || "low");
    return `<div class="metric-big">
      <span class="metric-num ${cls}">${payload.abnormality_probability}</span>
      <span class="metric-label">${payload.calibrated === false ? "ABNORMALITY SCORE" : "P(ABNORMAL)"} · τ=${payload.threshold} · ${esc((payload.band || "").toUpperCase())} · ${esc((payload.prediction || "").toUpperCase())}</span>
    </div>`;
  }
  if (toolId === "segmentation") {
    const rows = (payload.structures || []).map((s) =>
      `<div class="metric-row"><span>STRUCTURE ${s.index}</span><span>${s.area_mm2 != null ? s.area_mm2 + " mm²" : s.area_px + " px (uncalibrated)"} · IOU ${s.score}</span></div>`
    ).join("");
    return `<div class="metric-list">${rows}<div class="metric-row"><span>CALIBRATION</span><span>${esc(payload.calibration || "")}</span></div></div>`;
  }
  if (toolId === "rag") {
    const matches = payload.matches || [];
    if (!matches.length) {
      return `<div class="metric-list"><div class="metric-row"><span>${esc((payload.note || "NO MATCHES").toUpperCase())}</span></div></div>`;
    }
    const rows = matches.map((m) =>
      `<div class="metric-row"><span>${esc(m.case_id)}</span><span>${m.similarity_percent}% · ${esc(m.region.toUpperCase())}</span></div>`
    ).join("");
    return `<div class="metric-list">${rows}<div class="metric-row"><span>COLLECTION</span><span>${esc(payload.collection || "")}</span></div></div>`;
  }
  const rows = Object.entries(payload)
    .filter(([, v]) => typeof v !== "object")
    .map(([k, v]) => `<div class="metric-row"><span>${esc(k)}</span><span>${esc(String(v))}</span></div>`)
    .join("");
  return `<div class="metric-list">${rows}</div>`;
}

function appendToken(block, text) {
  const stream = block.stream;
  const typing = stream.querySelector(".typing");
  if (typing) typing.remove();
  stream.textContent += text;
  scrollBottom();
}

function markBlockError(block, message) {
  if (block.failed) return;
  block.failed = true;
  state.error = true;
  block.stream.querySelector(".typing")?.remove();
  block.tools?.querySelectorAll(".m3-progress").forEach(el => { el.hidden = true; });
  block.tools?.querySelectorAll(".ts-text").forEach(el => { if (el.textContent !== "COMPLETE") el.textContent = "INTERRUPTED"; });
  const request = state.lastRequest;
  const note = document.createElement("div");
  note.className = "error-note";
  note.setAttribute("role", "alert");
  note.innerHTML = `<div class="error-content"><h3>ANALYSIS INTERRUPTED</h3><p>${esc(message)}</p><div class="error-actions"><button class="retry-btn">RETRY</button><button class="edit-btn">EDIT MESSAGE</button></div></div><img src="/static/assets/error-skeleton.svg" alt="" width="126" height="126"/>`;
  note.querySelector(".retry-btn").addEventListener("click", () => {
    if (state.running || !request) return;
    note.remove();
    if (request.type === "upload") startAnalysis(request.file, request.text);
    else if (request.type === "analyze") analyzeCurrent(request.text);
    else ask(request.text);
  });
  note.querySelector(".edit-btn").addEventListener("click", () => {
    if (state.running || !request) return;
    $("#historyInput").value = request.text;
    state.records = [];
    if (request.type === "upload") state.pickedFile = request.file;
    updateAttachment();
    autosize({ target: $("#historyInput") });
    $("#historyInput").focus();
  });
  block.body.appendChild(note);
  scrollBottom();
}

function renderReport(block, data, sessionId) {
  const { report, verification, evidence } = data;
  block.stream.style.display = "none";
  const tri = report.triage_summary;
  const wrap = document.createElement("div");
  wrap.className = "report" + (verification && verification.ok ? "" : " flagged");

  const findingsHTML = (report.findings || []).map((f) => `
    <div class="finding">
      <span class="finding-led"></span>
      <div class="finding-body">
        <div class="finding-head">
          <strong>${esc(f.structure.toUpperCase())}</strong>
          ${f.confidence_band ? `<span class="chip">${esc(f.confidence_band.toUpperCase())}</span>` : ""}
          ${(f.evidence_ids || []).map((id) => `<span class="chip ev">${esc(id)}</span>`).join("")}
        </div>
        <div class="finding-obs">${mdLite(f.observation)}</div>
        ${(f.measurements || []).map((m) =>
          `<div class="finding-meas">${m.value} ${esc(m.unit)}<span class="chip ev">${esc(m.source)}</span></div>`
        ).join("")}
      </div>
    </div>`).join("");

  const recoHTML = (report.recommendations || []).map((r) => `<li>${mdLite(r)}</li>`).join("");
  const limHTML = (report.limitations || []).map((l) => `<li>${esc(l)}</li>`).join("");
  const verChip = !verification
    ? ""
    : verification.scope === "deterministic_metadata"
      ? (verification.ok
        ? '<span class="chip ok">METADATA CHECKED</span>'
        : `<span class="chip warn">${(verification.warnings || verification.violations || []).length} METADATA WARNING(S)</span>`)
    : verification.ok
      ? '<span class="chip ok">EVIDENCE VERIFIED</span>'
      : `<span class="chip warn">${verification.violations.length} UNVERIFIED CLAIM(S)</span>`;

  wrap.innerHTML = `
    <div class="report-head">
      ${gaugeHTML(tri?.abnormality_probability, tri?.band, tri?.threshold, tri?.calibrated)}
      <div class="report-head-meta">
        <div class="report-head-line">
          <span class="chip violet">${esc(((report.study && report.study.region) || "REGION N/A").toUpperCase())}</span>
          ${tri && tri.threshold != null ? `<span class="chip">τ ${tri.threshold}</span>` : ""}
          ${verChip}
        </div>
        <div class="report-impression-peek">${mdLite(report.impression)}</div>
      </div>
    </div>
    <div class="report-section"><h3>FINDINGS</h3>${findingsHTML || '<div class="finding-obs">NONE REPORTED</div>'}</div>
    <div class="report-section"><h3>IMPRESSION</h3><div class="finding-obs">${mdLite(report.impression)}</div></div>
    ${recoHTML ? `<div class="report-section"><h3>RECOMMENDATIONS</h3><ul class="reco-list">${recoHTML}</ul></div>` : ""}
    ${casesHTML(evidence)}
    ${viewerHTML(sessionId, evidence)}
    ${limHTML ? `<div class="report-section report-limitations"><h3>LIMITATIONS</h3><ul class="limitation-list">${limHTML}</ul></div>` : ""}`;

  block.body.appendChild(wrap);

  const overlay = wrap.querySelector(".viewer-overlay");
  const range = wrap.querySelector(".ov-range");
  const toggle = wrap.querySelector(".switch");
  if (overlay) {
    overlay.style.opacity = "0.6";
    if (range) range.addEventListener("input", () => (overlay.style.opacity = toggle && !toggle.checked ? "0" : range.value / 100));
    if (toggle) toggle.addEventListener("change", () => (overlay.style.opacity = toggle.checked ? (range ? range.value / 100 : 0.6) : "0"));
  }
  scrollBottom();
}

function gaugeHTML(p, band, threshold, calibrated) {
  const len = Math.PI * 52;
  const hasScore = typeof p === "number" && Number.isFinite(p);
  const value = Math.max(0, Math.min(1, Number(p) || 0));
  const th = Math.max(0, Math.min(1, Number(threshold) || 0.33));
  const cx = 60, cy = 64, r = 52;
  const point = (fraction, radius) => {
    const ang = Math.PI * (1 - fraction);
    return [cx + radius * Math.cos(ang), cy - radius * Math.sin(ang)];
  };
  let ticks = "";
  for (let i = 0; i <= 10; i++) {
    const f = i / 10;
    const long = i % 5 === 0;
    const [x1, y1] = point(f, r + 4);
    const [x2, y2] = point(f, r + (long ? 10 : 7));
    ticks += `<line class="gauge-tick" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/>`;
  }
  const [tx1, ty1] = point(th, r - 6);
  const [tx2, ty2] = point(th, r + 10);
  const id = uid();
  const color = BAND_COLOR[band] || "#fff";
  return `<div class="gauge">
    <svg viewBox="0 0 120 76" width="136" height="86">
      <defs>
        <linearGradient id="${id}" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#aaa"/>
          <stop offset="1" stop-color="${color}"/>
        </linearGradient>
      </defs>
      ${ticks}
      ${threshold != null ? `<line class="gauge-th" x1="${tx1.toFixed(1)}" y1="${ty1.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}"/>` : ""}
      <path class="gauge-track" d="M8 64 A52 52 0 0 1 112 64"/>
      <path class="gauge-fill" d="M8 64 A52 52 0 0 1 112 64" stroke="url(#${id})" style="stroke-dasharray:${len.toFixed(1)};stroke-dashoffset:${(len * (1 - value)).toFixed(1)}"/>
      <text x="60" y="52" class="gauge-num">${hasScore ? Number(p).toFixed(2) : "—"}</text>
    </svg>
    <div class="gauge-label">${!hasScore ? "NO AUXILIARY SCORE" : calibrated === true ? "P(ABNORMAL)" : "ABNORMALITY SCORE"}${band ? " · " + esc(band.toUpperCase()) : ""}</div>
  </div>`;
}

function viewerHTML(sessionId, evidence) {
  const artifacts = (evidence || []).flatMap((e) => e.artifacts || []);
  const overlay = artifacts.find((a) => a.kind === "overlay");
  if (!sessionId || !overlay) return "";
  return `<div class="report-section"><h3>EVIDENCE VIEWER</h3>
    <div class="viewer-frame">
      <img src="/a/${esc(sessionId)}/preview.png" alt="radiograph"/>
      <img class="viewer-overlay" src="${esc(overlay.url)}" alt="auxiliary model overlay"/>
    </div>
    <div class="viewer-ctl">
      <label><input type="checkbox" class="switch" checked/> OVERLAY</label>
      <input type="range" class="ov-range" min="0" max="100" value="60" aria-label="Overlay opacity"/>
    </div>
  </div>`;
}

function casesHTML(evidence) {
  const rag = (evidence || []).find((e) => e.tool === "rag");
  if (!rag || !(rag.payload && rag.payload.matches) || !rag.payload.matches.length) return "";
  const refs = (rag.artifacts || []).filter((a) => a.kind === "reference");
  const cards = rag.payload.matches.map((m, i) => `
    <div class="case">
      ${refs[i] ? `<img src="${esc(refs[i].url)}" alt="reference case"/>` : ""}
      <div class="case-row"><span class="case-id">${esc(m.case_id)}</span><span class="case-sim">${m.similarity_percent}%</span></div>
      <div class="sim-bar"><i style="width:${m.similarity_percent}%"></i></div>
      <div class="case-ref">${esc(m.reference || "")}</div>
    </div>`).join("");
  return `<div class="report-section"><h3>REFERENCE CASES</h3><div class="cases">${cards}</div></div>`;
}

function renderStoredSession(rec) {
  addUserStudy(rec, rec.history);
  if (!rec.report) return;
  const block = addAssistantBlock();
  block.stream.style.display = "none";
  for (const ev of rec.evidence || []) {
    ensureToolCards(block);
    updateToolCard(block, { tool: ev.tool, status: "done", summary: ev.summary });
    fillToolCard(block, { tool: ev.tool, payload: ev.payload });
  }
  renderReport(block, { report: rec.report, verification: rec.verification, evidence: rec.evidence }, rec.id);
  if (Array.isArray(rec.turns)) {
    for (const turn of rec.turns) {
      if (turn.role === "user") {
        addUserText(turn.content);
      } else if (turn.role === "assistant") {
        const turnBlock = addAssistantBlock(true);
        turnBlock.done = true;
        const typing = turnBlock.stream.querySelector(".typing");
        if (typing) typing.remove();
        turnBlock.stream.textContent = turn.content;
      }
    }
  }
}

async function refreshSessions() {
  try {
    const data = await getJSON("/api/sessions");
    state.sessions = data.sessions || [];
    renderSessions();
  } catch (e) {}
}

function sessionTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function renderSessions() {
  const list = $("#sessionsList");
  const query = $("#sessionSearch").value.trim().toLowerCase();
  let sessions = state.sessions.filter(s => (!prefs.hideCompleted || !s.done) && String(s.title).toLowerCase().includes(query));
  sessions.sort((a, b) => {
    if (prefs.pinnedFirst) { const diff = Number(prefs.pins.includes(b.id)) - Number(prefs.pins.includes(a.id)); if (diff) return diff; }
    return prefs.sort === "oldest" ? a.ts - b.ts : b.ts - a.ts;
  });
  list.innerHTML = "";
  if (!sessions.length) {
    list.innerHTML = '<div class="sessions-empty">' + (state.sessions.length ? 'No cases match your filters.' : 'Your studies will appear here.') + '</div>';
    return;
  }
  const groups = new Map();
  for (const s of sessions) {
    const group = prefs.group === "status" ? (s.done ? "Completed" : "In progress") : prefs.group === "region" ? (s.region || "Unspecified body part") : "";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(s);
  }
  for (const [group, items] of groups) {
    if (group) { const label = document.createElement("div"); label.className = "eyebrow"; label.textContent = group.toUpperCase(); list.appendChild(label); }
    for (const s of items) {
      const wrap = document.createElement("div"); wrap.className = "session-wrap";
      const el = document.createElement("button"); el.className = "session menu-row"; el.dataset.id = s.id;
      el.innerHTML = '<span>' + esc(s.title) + '</span><small>' + esc(sessionTime(s.ts)) + ' · ' + (s.done ? 'completed' : 'in progress') + '</small>';
      el.addEventListener("click", () => loadSession(s.id));
      const pin = document.createElement("button"); pin.className = "pin-btn"; pin.textContent = prefs.pins.includes(s.id) ? "★" : "☆";
      pin.setAttribute("aria-label", (prefs.pins.includes(s.id) ? "Unpin " : "Pin ") + s.title);
      pin.setAttribute("aria-pressed", String(prefs.pins.includes(s.id)));
      pin.addEventListener("click", () => { prefs.pins = prefs.pins.includes(s.id) ? prefs.pins.filter(id => id !== s.id) : [...prefs.pins, s.id]; savePreferences(); });
      wrap.append(el, pin); list.appendChild(wrap);
    }
  }
  markActiveSession(state.current && state.current.id);
}
function markActiveSession(id) {
  document.querySelectorAll(".session").forEach(el => {
    el.classList.toggle("active", el.dataset.id === id);
    if (el.dataset.id === id) el.setAttribute("aria-current", "true"); else el.removeAttribute("aria-current");
  });
}
async function loadSession(id) {
  if (state.running || state.loading || state.readingFiles) return;
  clearSnackbar();
  state.loading = true;
  setBusy(true);
  let rec;
  try { rec = await getJSON("/api/sessions/" + encodeURIComponent(id)); }
  catch (e) { snackbar("SESSION UNAVAILABLE"); return; }
  finally { state.loading = false; setBusy(false); }
  closeDrawer();
  state.current = { id, reported: !!rec.report, title: rec.title, preview_url: rec.preview_url, history: rec.history || "" };
  state.error = false; state.lastRequest = null; state.lastStudyEl = null;
  clearComposer();
  $("#thread").innerHTML = "";
  showConversation();
  renderStoredSession(rec);
  markActiveSession(id); renderContext(); setBusy(false);
}

async function getJSON(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || "";
    } catch (e) {}
    throw new Error(detail || `${res.status}`);
  }
  return res.json();
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function mdLite(text) {
  let t = esc(text);
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\n/g, "<br/>");
  return t;
}

let snackbarTimer = null;
function clearSnackbar() {
  clearTimeout(snackbarTimer);
  $("#snackbar").classList.remove("show");
}
function snackbar(message) {
  const el = $("#snackbar");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(snackbarTimer);
  snackbarTimer = setTimeout(() => el.classList.remove("show"), 4000);
}

function scrollBottom() {
  const thread = $("#thread");
  thread.scrollTop = thread.scrollHeight;
}
