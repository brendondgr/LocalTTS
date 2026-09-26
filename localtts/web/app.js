/* LocalTTS web UI: a chat where you type text and get speech back. No framework, no build.
 *
 * Talks to a LocalTTS backend: this server by default, or another one (a remote GPU box
 * forwarded with `ssh -L`) chosen in Settings. AI-Enhance always runs on this server.
 * Chats live in IndexedDB; audio made with saving off is kept in memory only.
 */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const LOCAL = location.origin;
const SVG = {
  wave: '<svg viewBox="0 0 24 24"><path d="M4 10v4M8 6v12M12 9v6M16 4v16M20 10v4"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  redo: '<svg viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.3-5.7M20 4v5h-5"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>',
  edit: '<svg viewBox="0 0 24 24"><path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  spark: '<svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 3l10 18H2zM12 10v5M12 18h.01"/></svg>',
};

// ---------------------------------------------------------------- preferences

const PREFS_KEY = "localtts.prefs";
const prefs = Object.assign({
  backend: "local", backendUrl: "", voice: "k:af_heart", format: "wav", save: true, saveDir: "",
  seed: 42, cfg: "", speed: 1, style: "subtle", llmModel: "", chat: "",
}, (() => { try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; } })());
function savePrefs() { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* private mode */ } }

// ---------------------------------------------------------------- HTTP

function backendBase() {
  return prefs.backend === "custom" && prefs.backendUrl ? prefs.backendUrl.replace(/\/+$/, "") : LOCAL;
}
function backendName(base = backendBase()) {
  if (base === LOCAL) return "This machine";
  try { return new URL(base).host; } catch { return base; }
}

async function errorText(resp) {
  try {
    const j = await resp.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map(d => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
  } catch { /* not JSON */ }
  return `${resp.status} ${resp.statusText}`;
}

async function http(path, { base = backendBase(), json, form, method, signal, raw } = {}) {
  const opts = { method: method || (json || form ? "POST" : "GET"), signal };
  if (json) { opts.headers = { "content-type": "application/json" }; opts.body = JSON.stringify(json); }
  if (form) opts.body = form;
  let resp;
  try { resp = await fetch(base + path, opts); }
  catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error(`Can't reach ${backendName(base)} (${base}). Is LocalTTS running there?`);
  }
  if (!resp.ok) throw new Error(await errorText(resp));
  return raw ? resp : resp.json();
}

// ---------------------------------------------------------------- IndexedDB (chats + audio)

const db = {
  _p: null,
  open() {
    return this._p ||= new Promise((ok, fail) => {
      const r = indexedDB.open("localtts", 1);
      r.onupgradeneeded = () => { r.result.createObjectStore("chats", { keyPath: "id" }); r.result.createObjectStore("audio"); };
      r.onsuccess = () => ok(r.result);
      r.onerror = () => fail(r.error);
    });
  },
  async run(store, mode, fn) {
    const d = await this.open();
    return new Promise((ok, fail) => {
      const tx = d.transaction(store, mode);
      const req = fn(tx.objectStore(store));
      tx.oncomplete = () => ok(req && req.result);
      tx.onerror = () => fail(tx.error);
    });
  },
  all: s => db.run(s, "readonly", st => st.getAll()),
  get: (s, k) => db.run(s, "readonly", st => st.get(k)),
  put: (s, v, k) => db.run(s, "readwrite", st => (k === undefined ? st.put(v) : st.put(v, k))),
  del: (s, k) => db.run(s, "readwrite", st => st.delete(k)),
  clear: s => db.run(s, "readwrite", st => st.clear()),
};
const safe = p => p.catch(e => { console.warn("storage:", e); });

// ---------------------------------------------------------------- state

const state = {
  chats: [],            // [{id, title, created, updated, turns: [...]}]
  chat: null,           // current chat
  voices: { custom: [], kokoro: [] },
  health: null,
  tags: { tags: [], experimental: [] },
  memAudio: new Map(),  // turn id -> Blob (saving off: never written anywhere)
  urls: new Map(),      // turn id -> object URL
  pending: new Map(),   // turn id -> {ctrl, timer}
  autoplay: new Set(),  // turn ids to play once when their audio appears
  undo: null,           // draft before the last Enhance
};
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

// ---------------------------------------------------------------- helpers

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function knownTag(word) {
  const w = word.toLowerCase().replace(/\s+/g, " ").trim();
  return [...state.tags.tags, ...state.tags.experimental].some(t => t.tag === w);
}
function withTags(text) {  // escaped HTML with (tag) as pills
  return esc(text).replace(/[(\[]\s*([A-Za-z][A-Za-z ]{0,30}?)\s*[)\]]/g, (m, w) => (knownTag(w) ? `<span class="tag">${esc(w.toLowerCase())}</span>` : m));
}
function fmtS(s) { return s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`; }
let toastTimer;
function toast(msg, ms = 2600) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}
function kokoroLabel(v) {
  const name = v.split("_")[1] || v;
  return name.charAt(0).toUpperCase() + name.slice(1);
}
const KOKORO_GROUPS = { af: "Kokoro · US female", am: "Kokoro · US male", bf: "Kokoro · UK female", bm: "Kokoro · UK male" };

function parseVoice(value) {  // "c:Name" | "d:" | "k:af_heart"
  const [k, ...rest] = (value || "").split(":");
  const name = rest.join(":");
  if (k === "c") return { kind: "custom", name, engine: "breeze", label: name };
  if (k === "d") return { kind: "design", name: "", engine: "breeze", label: "Designed voice" };
  return { kind: "kokoro", name: name || "af_heart", engine: "kokoro", label: kokoroLabel(name || "af_heart") };
}
const currentVoice = () => parseVoice($("#voice").value);

// ---------------------------------------------------------------- backend status

async function refreshHealth() {
  const dot = $("#backend-dot"), label = $("#backend-label");
  try {
    state.health = await http("/health");
    $("#brand-sub").textContent = `speech API · v${state.health.version}`;
    dot.className = "dot ok";
    label.textContent = backendName();
  } catch (e) {
    state.health = null;
    dot.className = "dot err";
    label.textContent = `${backendName()} · offline`;
  }
  renderEngines();
  return state.health;
}

function engineState(name) { return state.health?.engines?.[name]?.state || "unknown"; }

function renderEngines() {
  const box = $("#engines"), ctl = $("#engine-controls");
  const h = state.health;
  box.innerHTML = ""; ctl.innerHTML = "";
  if (!h) { ctl.innerHTML = '<p class="muted small">Backend offline.</p>'; return; }
  $("#idle-note").textContent = `· unload after ${Math.round(h.idle_unload_s / 60)} min idle`;
  $("#save-where").textContent = h.outputs || "~/Music/TTS";
  $("#opt-save-dir").placeholder = h.outputs ? h.outputs.replace(/^\/home\/[^/]+/, "~") : "default";
  for (const [name, e] of Object.entries(h.engines)) {
    const cls = e.state === "ready" ? "ok" : e.state === "unloaded" ? "" : "busy";
    const idle = e.unloads_in_s != null ? `, unloads in ${Math.ceil(e.unloads_in_s / 60)} min` : "";
    box.insertAdjacentHTML("beforeend",
      `<span class="engine-chip" title="${esc(name)}: ${esc(e.state)}${esc(idle)}"><span class="dot ${cls}"></span>${esc(name[0].toUpperCase() + name.slice(1))}</span>`);
    const row = document.createElement("div");
    row.className = "engine-row";
    row.innerHTML = `<span class="dot ${cls}"></span><b>${esc(name)}</b><span class="grow">${esc(e.state)}${esc(idle)}</span>
      <button class="btn" data-load="${esc(name)}">Load</button><button class="btn" data-unload="${esc(name)}">Unload</button>`;
    row.querySelector("[data-load]").disabled = e.state !== "unloaded";
    row.querySelector("[data-unload]").disabled = e.state === "unloaded";
    ctl.append(row);
  }
}

async function engineAction(btn) {
  const load = btn.dataset.load, name = load || btn.dataset.unload;
  btn.disabled = true; btn.textContent = load ? "Loading…" : "Unloading…";
  try {
    await http(`/v1/${load ? "load" : "unload"}?engine=${encodeURIComponent(name)}`, { method: "POST" });
    toast(`${name} ${load ? "loaded" : "unloaded"}`);
  } catch (e) { toast(e.message, 5000); }
  await refreshHealth();
}

// ---------------------------------------------------------------- voices

async function refreshVoices() {
  try { state.voices = await http("/v1/voices"); }
  catch { state.voices = { custom: [], kokoro: state.voices.kokoro || [] }; }
  renderVoiceSelect();
  renderVoiceList();
}

function renderVoiceSelect() {
  const sel = $("#voice");
  const want = prefs.voice;
  const groups = [];
  const custom = state.voices.custom || [];
  groups.push(`<optgroup label="Your voices · Breeze">${custom.length
    ? custom.map(v => `<option value="c:${esc(v.name)}">${esc(v.name)}</option>`).join("")
    : '<option value="" disabled>None yet: add one in Voices</option>'}</optgroup>`);
  groups.push('<optgroup label="Breeze"><option value="d:">Describe a voice…</option></optgroup>');
  const byGroup = {};
  for (const v of state.voices.kokoro || []) (byGroup[v.slice(0, 2)] ||= []).push(v);
  for (const [g, list] of Object.entries(byGroup)) {
    groups.push(`<optgroup label="${esc(KOKORO_GROUPS[g] || "Kokoro")}">${list.map(v =>
      `<option value="k:${esc(v)}">${esc(kokoroLabel(v))}</option>`).join("")}</optgroup>`);
  }
  sel.innerHTML = groups.join("");
  const values = [...sel.options].map(o => o.value);
  sel.value = values.includes(want) ? want : (values.includes("k:af_heart") ? "k:af_heart" : values.find(Boolean) || "");
  applyVoice(values.includes(want));
}

let lastKind = null;
function applyVoice(remember = true) {  // remember=false: a fallback, keep the saved choice
  const v = currentVoice();
  if (remember) { prefs.voice = $("#voice").value; savePrefs(); }
  const breeze = v.engine === "breeze";
  const dirRow = $("#direction-row"), dir = $("#direction");
  // A voice description and a delivery direction share the field; don't carry one into the other.
  if (lastKind && lastKind !== v.kind && (lastKind === "design" || v.kind === "design")) {
    dir.value = ""; $("#btn-direction").classList.remove("on");
  }
  lastKind = v.kind;
  if (v.kind === "design") {
    dir.placeholder = "Describe the voice: e.g. a warm, calm young woman, slightly amused";
    dirRow.hidden = false;
  } else {
    dir.placeholder = "Direction: e.g. whisper, nervous and quick";
    if (!breeze) dirRow.hidden = true;
    else if (!dir.value) dirRow.hidden = !$("#btn-direction").classList.contains("on");
  }
  $("#btn-direction").disabled = !breeze || v.kind === "design";
  $("#btn-direction").classList.toggle("on", breeze && !dirRow.hidden);
  for (const id of ["#btn-tags", "#btn-enhance"]) {
    $(id).disabled = !breeze;
    $(id).title = breeze ? $(id).dataset.title || $(id).title : "Kokoro voices don't support audio tags";
  }
  document.querySelectorAll(".breeze-only").forEach(el => { el.hidden = !breeze; });
  document.querySelectorAll(".kokoro-only").forEach(el => { el.hidden = breeze; });
  updateSend();
}

function renderVoiceList() {
  const list = $("#voice-list");
  document.querySelectorAll(".backend-name").forEach(el => { el.textContent = backendName(); });
  const custom = state.voices.custom || [];
  if (!custom.length) {
    list.innerHTML = '<p class="muted small">No saved voices on this backend yet. Use <b>Add voice</b> to upload or record one.</p>';
    return;
  }
  list.innerHTML = "";
  for (const v of custom) {
    const row = document.createElement("div");
    row.className = "voice-row";
    row.innerHTML = `
      <div class="name">${esc(v.name)}
        <span class="badge">${v.duration_s.toFixed(1)} s</span>
        <span class="badge">${v.source === "design" ? "designed" : "uploaded"}</span>
        ${v.transcript_auto ? '<span class="badge auto" title="Written by Whisper: check it">auto transcript</span>' : ""}</div>
      <div class="btns">
        <button class="btn" data-use>Use</button>
        <button class="icon-btn" data-play title="Play reference">${SVG.play}</button>
        <button class="icon-btn" data-edit title="Edit transcript">${SVG.edit}</button>
        <button class="icon-btn" data-del title="Delete permanently">${SVG.trash}</button>
      </div>
      <div class="transcript" title="${v.source === "design" ? esc("Designed from: " + (v.instruction || "")) : "Transcript"}">${esc(v.transcript)}</div>`;
    row.querySelector("[data-use]").onclick = () => {
      $("#voice").value = `c:${v.name}`; applyVoice(); $("#voices-dialog").close(); $("#text").focus();
    };
    row.querySelector("[data-play]").onclick = () => {
      let a = row.querySelector("audio");
      if (!a) {
        a = document.createElement("audio"); a.controls = true;
        a.src = `${backendBase()}/v1/voices/${encodeURIComponent(v.name)}/audio`;
        row.append(a);
      }
      a.play();
    };
    const tr = row.querySelector(".transcript"), editBtn = row.querySelector("[data-edit]");
    editBtn.onclick = async () => {
      if (tr.contentEditable !== "true") {
        tr.contentEditable = "true"; tr.focus(); editBtn.innerHTML = SVG.check; editBtn.title = "Save transcript";
        return;
      }
      try {
        await http(`/v1/voices/${encodeURIComponent(v.name)}`, { method: "PATCH", json: { transcript: tr.textContent.trim() } });
        toast("Transcript saved");
        await refreshVoices();
      } catch (e) { toast(e.message, 5000); }
    };
    row.querySelector("[data-del]").onclick = async () => {
      if (!confirm(`Permanently delete the voice "${v.name}"? This cannot be undone.`)) return;
      try {
        await http(`/v1/voices/${encodeURIComponent(v.name)}`, { method: "DELETE" });
        toast(`Deleted ${v.name}`);
        await refreshVoices();
      } catch (e) { toast(e.message, 5000); }
    };
    list.append(row);
  }
}

// add voice: file, drop or record
let addAudio = null;  // {blob, name}
function setAddAudio(blob, name) {
  addAudio = { blob, name };
  $("#file-label").textContent = name;
  const p = $("#add-preview");
  if (p.src) URL.revokeObjectURL(p.src);
  p.src = URL.createObjectURL(blob); p.hidden = false;
}
let recorder = null, recTimer = null;
async function toggleRecord() {
  const btn = $("#btn-record"), label = $("#record-label");
  if (recorder) { recorder.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true } }); }
  catch (e) { toast(`Microphone unavailable: ${e.message}`, 5000); return; }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = e => chunks.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    clearInterval(recTimer); btn.classList.remove("recording"); label.textContent = "Record";
    const type = recorder.mimeType || "audio/webm";
    setAddAudio(new Blob(chunks, { type }), `recording.${type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm"}`);
    recorder = null;
  };
  recorder.start();
  const t0 = Date.now();
  btn.classList.add("recording");
  const tick = () => {
    const s = (Date.now() - t0) / 1000;
    label.textContent = `Stop · ${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    if (s >= 60) recorder?.stop();
  };
  tick(); recTimer = setInterval(tick, 250);
}

async function submitAddVoice(e) {
  e.preventDefault();
  const f = e.target, status = $("#add-status"), btn = f.querySelector("[type=submit]");
  if (!addAudio) { status.className = "status small err"; status.textContent = "Choose, drop or record a clip first."; return; }
  const fd = new FormData();
  fd.append("name", f.name.value.trim());
  fd.append("audio", addAudio.blob, addAudio.name);
  fd.append("transcript", f.transcript.value.trim());
  fd.append("overwrite", f.overwrite.checked ? "true" : "false");
  btn.disabled = true; status.className = "status small";
  status.textContent = f.transcript.value.trim() ? "Saving…" : "Saving and transcribing with Whisper…";
  try {
    const v = await http("/v1/voices", { form: fd });
    status.className = "status small ok";
    status.textContent = `Saved "${v.name}" (${v.duration_s.toFixed(1)} s).` +
      (v.transcript_auto ? ` Whisper heard: "${v.transcript}". Fix it in Saved if it's wrong.` : "") +
      (v.warning ? ` ${v.warning}.` : "");
    prefs.voice = `c:${v.name}`;
    f.reset(); addAudio = null; $("#file-label").textContent = "or drop a file here"; $("#add-preview").hidden = true;
    await refreshVoices();
  } catch (err) { status.className = "status small err"; status.textContent = err.message; }
  btn.disabled = false;
}

async function submitDesign(e) {
  e.preventDefault();
  const f = e.target, status = $("#design-status"), btn = f.querySelector("[type=submit]");
  btn.disabled = true; status.className = "status small";
  status.textContent = engineState("breeze") === "ready" ? "Designing…" : "Designing… (Breeze is loading onto the GPU first, ~30 s)";
  try {
    const v = await http("/v1/voices/design", { json: { name: f.name.value.trim(), instruction: f.instruction.value.trim(), overwrite: f.overwrite.checked } });
    status.className = "status small ok";
    status.textContent = `Saved "${v.name}". It's selected; listen to its sample under Saved.`;
    prefs.voice = `c:${v.name}`;
    f.reset();
    await refreshVoices();
  } catch (err) { status.className = "status small err"; status.textContent = err.message; }
  btn.disabled = false;
  refreshHealth();
}

// ---------------------------------------------------------------- tags & enhance

async function loadTags() {
  try { state.tags = await http("/v1/enhance/tags", { base: LOCAL }); } catch { /* older server */ }
  const menu = $("#tags-menu");
  const item = t => `<button type="button" class="menu-item" data-tag="${esc(t.tag)}"><code>(${esc(t.tag)})</code><span>${esc(t.description)}</span></button>`;
  menu.innerHTML = `<h5>Breeze audio tags</h5>${state.tags.tags.map(item).join("")}` +
    (state.tags.experimental.length ? `<h5>Experimental · try it</h5>${state.tags.experimental.map(item).join("")}` : "");
}

function insertTag(tag) {
  const ta = $("#text");
  const s = ta.selectionStart ?? ta.value.length, e = ta.selectionEnd ?? s;
  const before = ta.value.slice(0, s), after = ta.value.slice(e);
  const pre = before && !/\s$/.test(before) ? " " : "";
  const post = after && /^\s/.test(after) ? "" : " ";
  const ins = `${pre}(${tag})${post}`;
  ta.value = before + ins + after;
  ta.focus(); ta.selectionStart = ta.selectionEnd = s + ins.length;
  autosize(); updateSend();
  $("#tags-menu").hidden = true;
}

function showNotice(html, warn = false) {
  const n = $("#enhance-notice");
  n.className = "notice" + (warn ? " warn" : "");
  n.innerHTML = html; n.hidden = false;
}
const hideNotice = () => { $("#enhance-notice").hidden = true; };

async function enhanceDraft() {
  const ta = $("#text"), dir = $("#direction"), btn = $("#btn-enhance");
  const text = ta.value.trim();
  if (!text) { toast("Type something first"); return; }
  const v = currentVoice();
  btn.classList.add("loading"); btn.disabled = true;
  try {
    const r = await http("/v1/enhance", {
      base: LOCAL,
      json: { text, style: prefs.style, model: prefs.llmModel || null, direction: dir.value.trim() || null },
    });
    state.undo = { text: ta.value, direction: dir.value };
    ta.value = r.text; autosize();
    let applied = false, suggestion = "";
    if (r.delivery) {
      if (v.kind === "custom" && !dir.value.trim()) {
        dir.value = r.delivery; $("#direction-row").hidden = false; $("#btn-direction").classList.add("on"); applied = true;
      } else if (!dir.value.toLowerCase().includes(r.delivery.toLowerCase())) {
        suggestion = r.delivery;
      }
    }
    const what = r.tags_added ? `Added ${r.tags_added} tag${r.tags_added > 1 ? "s" : ""}` : "No tags fit";
    showNotice(`${r.warning ? SVG.warn : SVG.spark}<span>${esc(what)}${applied ? ` and a direction: <i>${esc(r.delivery)}</i>` : ""}.` +
      `${suggestion ? ` Suggested ${v.kind === "design" ? "delivery" : "direction"}: <i>${esc(suggestion)}</i>` : ""}` +
      `${r.warning ? ` ${esc(r.warning[0].toUpperCase() + r.warning.slice(1))}.` : ""}</span>` +
      `${suggestion ? '<button type="button" data-action="use-suggestion">Use it</button>' : ""}` +
      '<button type="button" data-action="undo-enhance">Undo</button><button type="button" data-action="hide-notice">Dismiss</button>', !!r.warning);
    $("#enhance-notice").dataset.suggestion = suggestion;
  } catch (e) {
    showNotice(`${SVG.warn}<span>AI-Enhance failed: ${esc(e.message)}${/unreachable|503|healthy/.test(e.message)
      ? " Start a model on this machine, e.g. <code>lcpp start gemma4-E4B</code>, or pick one in Settings." : ""}</span>` +
      '<button type="button" data-action="hide-notice">Dismiss</button>', true);
  }
  btn.classList.remove("loading"); btn.disabled = false;
  updateSend();
}

async function loadLlmModels() {
  const sel = $("#llm-model"), status = $("#llm-status");
  try {
    const r = await http("/v1/enhance/models", { base: LOCAL });
    $("#llm-url").textContent = `· ${r.url}`;
    sel.innerHTML = `<option value="">Default (${esc(r.default)})</option>` + r.models.map(m =>
      `<option value="${esc(m.id)}">${esc(m.id)}${m.healthy ? "" : " (offline)"}</option>`).join("");
    status.textContent = "";
  } catch (e) {
    sel.innerHTML = '<option value="">Default</option>';
    status.className = "status small err"; status.textContent = e.message;
  }
  sel.value = [...sel.options].some(o => o.value === prefs.llmModel) ? prefs.llmModel : "";
}

// ---------------------------------------------------------------- chats

function newChat() {
  const c = { id: uid(), title: "New chat", created: Date.now(), updated: Date.now(), turns: [] };
  return c;
}

async function persist(chat) {
  if (!chat.turns.length) return;
  chat.updated = Date.now();
  const clean = { ...chat, turns: chat.turns.map(t => ({ ...t })) };
  await safe(db.put("chats", clean));
  if (!state.chats.includes(chat)) state.chats.unshift(chat);
  if (state.chat === chat && prefs.chat !== chat.id) { prefs.chat = chat.id; savePrefs(); }
  renderChatList();
}

function openChat(chat) {
  state.chat = chat;
  prefs.chat = chat.turns.length ? chat.id : ""; savePrefs();
  $("#chat-title").textContent = chat.title;
  renderThread();
  renderChatList();
  document.getElementById("app").classList.remove("side-open");
}

async function deleteChat(chat) {
  if (!confirm(`Delete "${chat.title}"? Audio saved on the backend stays in its folder.`)) return;
  for (const t of chat.turns) { await safe(db.del("audio", t.id)); state.memAudio.delete(t.id); dropUrl(t.id); }
  await safe(db.del("chats", chat.id));
  state.chats = state.chats.filter(c => c !== chat);
  if (state.chat === chat) openChat(newChat()); else renderChatList();
}

function renderChatList() {
  const nav = $("#chat-list");
  nav.innerHTML = "";
  const day = 864e5, today = new Date().setHours(0, 0, 0, 0);
  const groups = [["Today", today], ["Yesterday", today - day], ["Previous 7 days", today - 7 * day], ["Older", -Infinity]];
  const sorted = [...state.chats].sort((a, b) => b.updated - a.updated);
  let gi = -1;
  for (const c of sorted) {
    const g = groups.findIndex(([, from]) => c.updated >= from);
    if (g !== gi) { gi = g; nav.insertAdjacentHTML("beforeend", `<div class="chat-group">${groups[g][0]}</div>`); }
    const item = document.createElement("div");
    item.className = "chat-item" + (c === state.chat ? " active" : "");
    item.innerHTML = `<button class="open" title="${esc(c.title)}">${esc(c.title)}</button>
      <button class="icon-btn small del" title="Delete chat" aria-label="Delete chat">${SVG.trash}</button>`;
    item.querySelector(".open").onclick = () => openChat(c);
    item.querySelector(".del").onclick = () => deleteChat(c);
    nav.append(item);
  }
}

// ---------------------------------------------------------------- thread rendering

function dropUrl(id) { const u = state.urls.get(id); if (u) { URL.revokeObjectURL(u); state.urls.delete(id); } }

async function audioUrl(turn) {
  if (state.urls.has(turn.id)) return state.urls.get(turn.id);
  let blob = state.memAudio.get(turn.id);
  if (!blob && turn.kept) blob = await safe(db.get("audio", turn.id));
  if (!blob) return null;
  const u = URL.createObjectURL(blob);
  state.urls.set(turn.id, u);
  return u;
}

function renderThread() {
  const turns = $("#turns");
  turns.innerHTML = "";
  $("#empty").hidden = !!state.chat.turns.length;
  for (const t of state.chat.turns) turns.append(turnEl(t));
  scrollDown();
}

function scrollDown() { const th = $("#thread"); th.scrollTop = th.scrollHeight; }

function turnEl(t) {
  const el = document.createElement("div");
  el.className = "turn"; el.dataset.id = t.id;
  const dirLabel = t.voice.kind === "design" ? "Voice" : "Direction";
  el.innerHTML = `
    <div class="msg-user">
      <div class="bubble">${withTags(t.text)}</div>
      ${t.direction ? `<div class="msg-meta-user"><span class="dir" title="${dirLabel}">${esc(t.direction)}</span></div>` : ""}
    </div>
    <div class="msg-tts">
      <div class="avatar">${SVG.wave}</div>
      <div class="tts-body">
        <div class="tts-head"><b>${esc(t.voiceLabel)}</b><span class="muted">${esc(t.engine)}${t.backend !== LOCAL ? " · " + esc(backendName(t.backend)) : ""}</span></div>
        <div class="slot"></div>
      </div>
    </div>`;
  fillSlot(el, t);
  return el;
}

function fillSlot(el, t) {
  const slot = el.querySelector(".slot");
  if (t.status === "pending") {
    const cold = !["ready", "busy"].includes(engineState(t.engine));
    slot.innerHTML = `<div class="card pending"><div class="wave"><i></i><i></i><i></i><i></i><i></i></div>
      <div class="grow"><div>Generating… <span class="elapsed">0 s</span></div>
      ${cold ? `<div class="hint">${t.engine === "breeze" ? "Loading Breeze onto the GPU first (~30 s)" : "Loading Kokoro (a few seconds)"}</div>` : ""}</div>
      <button class="icon-btn small" title="Cancel" data-cancel>${SVG.x}</button></div>`;
    slot.querySelector("[data-cancel]").onclick = () => state.pending.get(t.id)?.ctrl.abort();
    return;
  }
  if (t.status === "error") {
    slot.innerHTML = `<div class="card error-card"><span>${esc(t.error)}</span><button class="btn" data-retry>Retry</button></div>`;
    slot.querySelector("[data-retry]").onclick = () => retry(t);
    return;
  }
  const m = t.meta || {};
  const parts = [];
  if (m.duration) parts.push(`${fmtS(m.duration)} audio`);
  if (m.gen_s) parts.push(`made in ${fmtS(m.gen_s)}`);
  if (t.engine === "breeze" && t.params?.seed != null) parts.push(`seed ${t.params.seed}`);
  parts.push(m.saved && m.saved !== "no"
    ? `<span class="saved" title="${esc(m.path || m.saved)}">saved · ${esc(m.path ? m.path.replace(/^\/home\/[^/]+/, "~") : m.saved)}</span>`
    : '<span class="unsaved">not saved</span>');
  slot.innerHTML = `<div class="card"><div class="audio-slot"></div></div>
    <div class="tts-meta">${parts.map(p => `<span>${p}</span>`).join("")}</div>
    <div class="tts-actions">
      <a class="icon-btn" data-dl title="Download" aria-label="Download">${SVG.download}</a>
      <button class="icon-btn" data-regen title="Regenerate with a new seed" aria-label="Regenerate">${SVG.redo}</button>
      <button class="icon-btn" data-reuse title="Edit and send again" aria-label="Edit">${SVG.edit}</button>
      <button class="icon-btn" data-copy title="Copy text" aria-label="Copy text">${SVG.copy}</button>
    </div>`;
  const aslot = slot.querySelector(".audio-slot"), dl = slot.querySelector("[data-dl]");
  audioUrl(t).then(u => {
    if (!u) {
      aslot.innerHTML = `<div class="gone">${m.saved && m.saved !== "no" ? "Audio isn't cached in this browser; it's saved on the backend." : "Audio was not saved, so it's gone after a reload."}</div>`;
      dl.hidden = true; return;
    }
    const a = document.createElement("audio");
    a.controls = true; a.preload = "metadata"; a.src = u;
    aslot.append(a);
    if (state.autoplay.delete(t.id)) a.play().catch(() => {});
    dl.href = u; dl.download = m.saved && m.saved !== "no" ? m.saved : `${(t.voiceLabel || "speech").toLowerCase().replace(/\W+/g, "-")}-${t.id}.${m.format || "wav"}`;
  });
  slot.querySelector("[data-regen]").onclick = () => {
    const params = { ...t.params };
    if (t.engine === "breeze") params.seed = Math.floor(Math.random() * 1e6);
    submitTurn({ ...t, params });
  };
  slot.querySelector("[data-reuse]").onclick = () => {
    $("#text").value = t.text; $("#direction").value = t.direction || "";
    const val = t.voice.kind === "custom" ? `c:${t.voice.name}` : t.voice.kind === "design" ? "d:" : `k:${t.voice.name}`;
    if ([...$("#voice").options].some(o => o.value === val)) $("#voice").value = val;
    if (t.direction) { $("#direction-row").hidden = false; $("#btn-direction").classList.add("on"); }
    applyVoice(); autosize(); $("#text").focus();
  };
  slot.querySelector("[data-copy]").onclick = async () => {
    try { await navigator.clipboard.writeText(t.text); toast("Copied"); } catch { toast("Copy failed"); }
  };
}

function refreshTurn(t) {
  const el = document.querySelector(`.turn[data-id="${t.id}"]`);
  if (el && state.chat.turns.includes(t)) fillSlot(el, t);
}

// ---------------------------------------------------------------- generate

function buildRequest() {
  const v = currentVoice();
  const text = $("#text").value.trim();
  const direction = $("#direction").value.trim();
  const p = { text, format: prefs.format, no_save: !prefs.save };
  if (prefs.save && prefs.saveDir.trim()) p.save_dir = prefs.saveDir.trim();
  if (v.kind === "kokoro") {
    Object.assign(p, { name: v.name, engine: "kokoro", speed: Number(prefs.speed) || 1 });
  } else {
    Object.assign(p, { engine: "breeze", seed: Number.isFinite(+prefs.seed) ? +prefs.seed : 42 });
    if (v.kind === "custom") p.name = v.name;
    if (direction) p.instruction = direction;
    if (prefs.cfg !== "" && +prefs.cfg > 0) p.cfg_scale = +prefs.cfg;
  }
  return { v, text, direction: v.engine === "breeze" ? direction : "", params: p };
}

function updateSend() {
  const v = currentVoice();
  const ok = $("#text").value.trim() && (v.kind !== "design" || $("#direction").value.trim());
  $("#btn-send").disabled = !ok;
  $("#btn-send").title = v.kind === "design" && !$("#direction").value.trim() ? "Describe the voice first" : "Generate (Enter)";
}

async function onSubmit(e) {
  e?.preventDefault();
  if ($("#btn-send").disabled) return;
  const { v, text, direction, params } = buildRequest();
  const turn = { id: uid(), text, direction, voice: { kind: v.kind, name: v.name }, voiceLabel: v.label,
                 engine: v.engine, params, backend: backendBase(), created: Date.now() };
  $("#text").value = ""; autosize(); hideNotice(); state.undo = null;
  if (v.kind !== "design") { $("#direction").value = ""; if (v.kind === "custom") { $("#direction-row").hidden = true; $("#btn-direction").classList.remove("on"); } }
  updateSend();
  submitTurn(turn);
}

async function submitTurn(src) {
  const chat = state.chat;
  const t = { ...src, id: src.id && !chat.turns.some(x => x.id === src.id) ? src.id : uid(),
              status: "pending", meta: null, error: null, kept: false, created: Date.now(), backend: src.backend || backendBase() };
  if (!chat.turns.length) {
    const plain = t.text.replace(/[(\[][a-z ]+[)\]]\s*/gi, "").replace(/\s+/g, " ").trim();
    chat.title = (plain.length > 60 ? plain.slice(0, 57).trimEnd() + "…" : plain) || "Speech";
    $("#chat-title").textContent = chat.title;
  }
  chat.turns.push(t);
  $("#empty").hidden = true;
  $("#turns").append(turnEl(t));
  scrollDown();
  persist(chat);

  const ctrl = new AbortController(), t0 = performance.now();
  const timer = setInterval(() => {
    const el = document.querySelector(`.turn[data-id="${t.id}"] .elapsed`);
    if (el) el.textContent = `${Math.floor((performance.now() - t0) / 1000)} s`;
  }, 500);
  state.pending.set(t.id, { ctrl, timer });
  try {
    const resp = await http("/v1/speech", { base: t.backend, json: t.params, signal: ctrl.signal, raw: true });
    const blob = await resp.blob();
    const h = k => resp.headers.get(k);
    t.meta = {
      duration: parseFloat(h("X-Audio-Duration")) || null, rtf: parseFloat(h("X-RTF")) || null,
      gen_s: parseFloat(h("X-Generation-Seconds")) || (performance.now() - t0) / 1000,
      saved: h("X-LocalTTS-Saved") || "no", path: h("X-LocalTTS-Path") ? decodeURIComponent(h("X-LocalTTS-Path")) : "",
      format: t.params.format,
    };
    if (h("X-LocalTTS-Voice") && t.voice.kind === "design") t.voiceLabel = "Designed voice";
    if (t.params.no_save) state.memAudio.set(t.id, blob);  // saving off: memory only
    else { await safe(db.put("audio", blob, t.id)); t.kept = true; state.memAudio.set(t.id, blob); }
    t.status = "done"; state.autoplay.add(t.id);
  } catch (e) {
    t.status = "error";
    t.error = e.name === "AbortError" ? "Cancelled." : e.message;
  }
  clearInterval(timer); state.pending.delete(t.id);
  refreshTurn(t);
  persist(chat);
  refreshHealth();
}

function retry(t) {
  const chat = state.chat;
  chat.turns = chat.turns.filter(x => x !== t);
  document.querySelector(`.turn[data-id="${t.id}"]`)?.remove();
  submitTurn({ ...t, id: null });
}

// ---------------------------------------------------------------- composer & menus

function autosize() {
  const ta = $("#text");
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, window.innerHeight * 0.4) + "px";
}

function bindOptions() {
  const bind = (id, key, get = el => el.value, set = (el, v) => { el.value = v; }) => {
    const el = $(id); set(el, prefs[key]);
    el.addEventListener("change", () => { prefs[key] = get(el); savePrefs(); syncOptions(); });
  };
  bind("#opt-save", "save", el => el.checked, (el, v) => { el.checked = v; });
  bind("#opt-save-dir", "saveDir");
  bind("#opt-format", "format");
  bind("#opt-seed", "seed");
  bind("#opt-cfg", "cfg");
  bind("#opt-speed", "speed");
  bind("#opt-style", "style");
  syncOptions();
}
function syncOptions() {
  $("#save-dir-row").hidden = !prefs.save;
  $("#foot").textContent = prefs.save
    ? "Enter to generate · Shift+Enter for a new line"
    : "Saving is off: audio stays in this tab only · Enter to generate";
}

function closeMenus(except) {
  for (const id of ["#tags-menu", "#options-menu"]) if (id !== except) $(id).hidden = true;
}

function setupSettings() {
  const radios = document.querySelectorAll('input[name="backend"]'), url = $("#backend-url");
  $("#local-url").textContent = LOCAL;
  radios.forEach(r => { r.checked = r.value === prefs.backend; });
  url.value = prefs.backendUrl;
  url.disabled = prefs.backend !== "custom";
  const change = async () => {
    const was = backendBase();
    prefs.backend = [...radios].find(r => r.checked)?.value || "local";
    prefs.backendUrl = url.value.trim();
    url.disabled = prefs.backend !== "custom";
    savePrefs();
    if (backendBase() !== was) { await refreshHealth(); await refreshVoices(); testBackend(); }
  };
  radios.forEach(r => r.addEventListener("change", change));
  url.addEventListener("change", change);
  $("#llm-model").addEventListener("change", e => { prefs.llmModel = e.target.value; savePrefs(); });
}

async function testBackend() {
  const s = $("#backend-status");
  const base = prefs.backend === "custom" ? $("#backend-url").value.trim().replace(/\/+$/, "") : LOCAL;
  if (!base) { s.className = "status small err"; s.textContent = "Enter the forwarded URL, e.g. http://localhost:5041"; return; }
  s.className = "status small"; s.textContent = "Testing…";
  try {
    const h = await http("/health", { base });
    const eng = Object.entries(h.engines).map(([n, e]) => `${n} ${e.state}`).join(", ");
    s.className = "status small ok";
    s.textContent = `Connected to LocalTTS ${h.version} · ${h.voices} saved voice${h.voices === 1 ? "" : "s"} · ${eng}`;
  } catch (e) { s.className = "status small err"; s.textContent = e.message; }
}

const SUGGESTIONS = [
  ["Say hello", "Hi! This voice was generated right here, on my own machine."],
  ["A sigh of relief", "(sigh) Finally. The build passed on the first try."],
  ["Narrate", "In the quiet hours before dawn, the whole city seems to hold its breath."],
  ["Try Enhance", "Well, I really did not expect that to work. Let's do it again, just to be sure."],
];

// ---------------------------------------------------------------- events

function wire() {
  const ta = $("#text");
  ta.addEventListener("input", () => { autosize(); updateSend(); });
  ta.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); onSubmit(); }
  });
  $("#direction").addEventListener("input", updateSend);
  $("#direction").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); onSubmit(); } });
  $("#composer").addEventListener("submit", onSubmit);
  $("#voice").addEventListener("change", applyVoice);
  $("#add-form").addEventListener("submit", submitAddVoice);
  $("#design-form").addEventListener("submit", submitDesign);
  $("#add-file").addEventListener("change", e => { const f = e.target.files[0]; if (f) setAddAudio(f, f.name); });
  const drop = $("#drop");
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", e => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0]; if (f) setAddAudio(f, f.name);
  });
  $("#btn-enhance").dataset.title = $("#btn-enhance").title;
  $("#btn-tags").dataset.title = $("#btn-tags").title;


  document.addEventListener("click", e => {
    const el = e.target.closest("[data-action],[data-tag],[data-sug],[data-load],[data-unload],.tab");
    if (!e.target.closest(".menu-wrap")) closeMenus();
    if (!el) return;
    if (el.dataset.tag) return insertTag(el.dataset.tag);
    if (el.dataset.sug) {
      ta.value = SUGGESTIONS[+el.dataset.sug][1]; autosize(); updateSend(); ta.focus();
      if (el.dataset.sug === "3") toast("Pick a Breeze voice, then press Enhance", 3500);
      return;
    }
    if (el.dataset.load || el.dataset.unload) return engineAction(el);
    if (el.classList.contains("tab")) {
      el.parentElement.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t === el));
      el.closest("dialog").querySelectorAll(".pane").forEach(p => { p.hidden = p.dataset.pane !== el.dataset.tab; });
      return;
    }
    switch (el.dataset.action) {
      case "new-chat": openChat(newChat()); ta.focus(); break;
      case "open-sidebar": $("#app").classList.add("side-open"); break;
      case "close-sidebar": $("#app").classList.remove("side-open"); break;
      case "open-voices": refreshVoices(); $("#voices-dialog").showModal(); break;
      case "open-settings":
        setupSettingsValues(); refreshHealth(); loadLlmModels(); testBackend(); $("#settings-dialog").showModal(); break;
      case "close-dialog": el.closest("dialog").close(); break;
      case "toggle-direction": {
        const row = $("#direction-row"); row.hidden = !row.hidden;
        el.classList.toggle("on", !row.hidden);
        if (!row.hidden) $("#direction").focus(); else $("#direction").value = "";
        updateSend(); break;
      }
      case "clear-direction":
        $("#direction").value = "";
        if (currentVoice().kind !== "design") { $("#direction-row").hidden = true; $("#btn-direction").classList.remove("on"); }
        updateSend(); break;
      case "toggle-tags": { const m = $("#tags-menu"); closeMenus("#tags-menu"); m.hidden = !m.hidden; break; }
      case "toggle-options": { const m = $("#options-menu"); closeMenus("#options-menu"); m.hidden = !m.hidden; break; }
      case "enhance": enhanceDraft(); break;
      case "undo-enhance":
        if (state.undo) {
          ta.value = state.undo.text; $("#direction").value = state.undo.direction;
          if (!state.undo.direction && currentVoice().kind === "custom") { $("#direction-row").hidden = true; $("#btn-direction").classList.remove("on"); }
          state.undo = null; autosize(); updateSend();
        }
        hideNotice(); break;
      case "use-suggestion": {
        const s = $("#enhance-notice").dataset.suggestion, d = $("#direction");
        d.value = d.value.trim() ? `${d.value.trim().replace(/[.\s]+$/, "")}. ${s}` : s;
        $("#direction-row").hidden = false; $("#btn-direction").classList.add("on");
        el.remove(); updateSend(); break;
      }
      case "hide-notice": hideNotice(); break;
      case "pick-file": $("#add-file").click(); break;
      case "record": toggleRecord(); break;
      case "test-backend": prefs.backendUrl = $("#backend-url").value.trim(); savePrefs(); testBackend(); refreshHealth().then(refreshVoices); break;
      case "clear-history":
        if (!confirm("Delete all chats and cached audio in this browser? Files saved on the backend are kept.")) break;
        safe(db.clear("chats")); safe(db.clear("audio"));
        state.chats = []; state.memAudio.clear(); [...state.urls.keys()].forEach(dropUrl);
        openChat(newChat()); toast("History cleared"); break;
    }
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeMenus(); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshHealth(); });
  setInterval(() => { if (!document.hidden) refreshHealth(); }, 10000);
}

function setupSettingsValues() {
  document.querySelectorAll('input[name="backend"]').forEach(r => { r.checked = r.value === prefs.backend; });
  $("#backend-url").value = prefs.backendUrl;
  $("#backend-url").disabled = prefs.backend !== "custom";
}

// ---------------------------------------------------------------- boot

(async function boot() {
  wire();
  setupSettings();
  bindOptions();
  autosize();
  await Promise.all([refreshHealth(), loadTags()]);
  $("#suggestions").innerHTML = SUGGESTIONS.map(([t, s], i) =>
    `<button type="button" class="suggestion" data-sug="${i}"><b>${esc(t)}</b>${withTags(s)}</button>`).join("");
  await refreshVoices();
  try {
    const chats = (await db.all("chats")) || [];
    for (const c of chats) for (const t of c.turns) if (t.status === "pending") { t.status = "error"; t.error = "Interrupted: the page was closed while generating."; }
    state.chats = chats;
  } catch (e) { console.warn("history unavailable:", e); }
  const last = state.chats.find(c => c.id === prefs.chat);
  openChat(last || newChat());
  $("#text").focus();
})();
