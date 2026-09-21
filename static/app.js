"use strict";

const $ = id => document.getElementById(id);
const audio = $("audio");
const state = { files: [], results: { automix: null, plain: null }, mode: "automix", busy: false, transition: 0, playing: false };
const colors = ["#ee937d", "#7fbbb0", "#b5b3d0", "#dcf87a", "#e7ba7e", "#a1bfd6"];
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const time = seconds => { const value = Math.max(0, Math.floor(Number(seconds) || 0)); return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`; };
const active = () => state.results[state.mode];

async function request(url, options) {
  const response = await fetch(url, options);
  let body;
  try { body = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}). Please retry.`); }
  if (!response.ok) {
    const detail = body.detail || body.error || body.message;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status}). Check the selected audio files and retry.`);
  }
  return body;
}

function showError(error) {
  $("error-banner").textContent = error instanceof Error ? error.message : String(error);
  $("error-banner").hidden = false;
  $("status-panel").hidden = true;
}

function setBusy(busy) {
  state.busy = busy;
  document.body.classList.toggle("busy", busy);
  ["build-upload", "hero-upload", "queue-upload", "browse-files", "reset-queue", "file-input"].forEach(id => $(id).disabled = busy);
  ["mode-automix", "mode-plain"].forEach(id => $(id).disabled = busy && !state.results[id.replace("mode-", "")]);
}

function setProgress(job, baseline = false) {
  $("status-panel").hidden = false;
  const value = Math.min(100, Math.max(0, Number(job.progress) || 0));
  const titles = { queued: "Your session is in the queue", downloading: "Bringing the tracks together", analyzing: "Finding the musical shape", planning: "Finding the right handoff", rendering: "Putting the pieces in motion", complete: "Your session is ready" };
  $("status-title").textContent = baseline ? "Preparing your A/B comparison" : (titles[job.status] || "Getting everything ready");
  $("status-message").textContent = job.message || (baseline ? "AutoMix is ready to play while the plain crossfade renders." : "Processing your audio locally.");
  $("status-percent").textContent = `${Math.round(value)}%`;
  $("progress-fill").style.width = `${value}%`;
  document.querySelectorAll("[data-step]").forEach(element => element.classList.toggle("current", element.dataset.step === job.status));
}

async function waitForJob(id, baseline = false) {
  const start = Date.now();
  while (Date.now() - start < 60 * 60 * 1000) {
    const job = await request(`/api/jobs/${encodeURIComponent(id)}`);
    setProgress(job, baseline);
    if (job.status === "error") throw new Error(job.error || job.message || "Audio processing failed. Please retry with another track.");
    if (job.status === "complete") return job;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error("This render is taking longer than expected. Reload to recover a completed session, or check the server logs.");
}

async function createMix(mode, baseline = false) {
  const form = new FormData();
  form.append("mode", mode);
  const source = state.results[mode === "plain" ? "automix" : "plain"];
  if (baseline && source) form.append("source_mix_id", source.id);
  else state.files.forEach(file => form.append("files", file));
  const started = await request("/api/mixes", { method: "POST", body: form });
  const job = await waitForJob(started.id, baseline);
  const id = job.result_id || job.mix_id || job.result?.id || started.id;
  const result = await request(`/api/mixes/${encodeURIComponent(id)}`);
  state.results[mode] = result;
  return result;
}

async function buildSession() {
  if (state.busy) return;
  if (state.files.length < 2 || state.files.length > 6) return showError("Choose 2–6 MP3, WAV, or FLAC files to build a mix.");
  $("error-banner").hidden = true;
  setBusy(true);
  setProgress({ status: "queued", progress: 0, message: "Preparing your tracks. The first analysis also loads the beat model and can take a few minutes." });
  try {
    // A new source must not accidentally reuse the previous session's A/B comparison.
    audio.pause();
    state.results = { automix: null, plain: null };
    state.mode = "automix";
    await createMix("automix");
    setMode("automix", false);
    $("mode-automix").disabled = false;
    await createMix("plain", true);
    if (state.mode === "plain") setMode("plain", false);
    $("status-panel").hidden = true;
  } catch (error) { showError(error); }
  finally { setBusy(false); }
}

function renderQueue() {
  const analyzed = (state.results.automix || state.results.plain)?.tracks || [];
  const restoredUploads = !state.files.length && analyzed.length;
  const tracks = state.files.length ? state.files.map((file, index) => ({ id: String(index), title: file.name.replace(/\.[^.]+$/, ""), artist: "Your local audio", size: file.size })) : restoredUploads ? analyzed.map(track => ({ ...track, artist: "Your local audio" })) : [];
  $("queue-count").textContent = String(tracks.length).padStart(2, "0");
  $("track-list").innerHTML = tracks.map((track, index) => {
    const analysis = analyzed[index];
    const bpm = analysis?.bpm ?? track.bpm;
    const meta = bpm ? `${Math.round(bpm)} BPM` : "Ready to analyze";
    return `<div class="track-row"><span class="track-number">${String(index + 1).padStart(2, "0")}</span><div class="track-art art-${index % 3}" aria-hidden="true"><span>${String(index + 1).padStart(2, "0")}</span></div><div class="track-info"><h3 title="${escapeHTML(track.title)}">${escapeHTML(track.title)}</h3><p>${escapeHTML(track.artist || "Local audio")}</p></div><div class="track-meta"><span>${time(analysis?.duration || track.duration)}</span><small class="${analysis ? "detected" : ""}">${escapeHTML(meta)}${bpm ? analysis ? " · detected" : "" : ""}</small></div></div>`;
  }).join("") || '<div class="loading-tracks">Add two tracks to start your session.</div>';
}

function chooseFiles(files) {
  if (state.busy) return;
  const selected = Array.from(files);
  if (selected.length < 2 || selected.length > 6) return showError("Select 2–6 files together. The selection becomes your complete, ordered queue.");
  if (selected.some(file => !/\.(mp3|wav|flac)$/i.test(file.name))) return showError("Please use MP3, WAV, or FLAC audio files.");
  if (selected.some(file => file.size > 100 * 1024 * 1024)) return showError("Each audio file must be 100 MB or smaller.");
  state.files = selected;
  audio.pause();
  state.results = { automix: null, plain: null };
  clearPlayer();
  $("error-banner").hidden = true;
  $("queue-actions").hidden = false;
  renderQueue();
  $("queue").scrollIntoView({ behavior: "smooth", block: "center" });
}

function clearPlayer() {
  audio.removeAttribute("src"); audio.load();
  $("waveform-empty").hidden = false;
  $("playhead").hidden = true;
  $("transition-labels").replaceChildren();
  ["play", "restart", "next-transition", "seek", "transition-select"].forEach(id => $(id).disabled = true);
  $("player-title").textContent = "A better in-between.";
  $("player-subtitle").textContent = "Your new queue is ready. Build a mix to listen.";
  $("mix-state").textContent = "AWAITING AUDIO"; $("mix-state").classList.remove("ready");
  $("duration").textContent = "0:00";
  $("out-title").textContent = "Current track"; $("in-title").textContent = "Next track";
  $("tier").textContent = "WAITING FOR ANALYSIS";
  ["cue-value", "overlap-value", "tempo-value"].forEach(id => $(id).textContent = "—");
  $("transition-explanation").textContent = "We look for a musical entry point, align the handoff, and let one track make room for the next.";
  $("inspector-detail").textContent = "Beat analysis · equal-power fades · bass handoff";
  ["export-audio", "export-plan"].forEach(id => { $(id).removeAttribute("href"); $(id).classList.add("disabled"); $(id).setAttribute("aria-disabled", "true"); });
  $("audition-buttons").innerHTML = '<button disabled>↗ Hear transition 01</button><button disabled>↗ Hear transition 02</button>';
  drawAll();
}

function positionForComparison(oldResult, newResult, currentTime) {
  if (!oldResult || !newResult) return 0;
  const transitions = oldResult.transitions || [];
  // Preserve the listener's position relative to the handoff, not the total mix duration.
  let index = transitions.findIndex(transition => currentTime >= transition.timeline_start - 15 && currentTime <= transition.timeline_end + 15);
  if (index < 0) index = Math.min(state.transition, transitions.length - 1);
  const previous = transitions[index];
  const next = newResult.transitions?.[index];
  if (!previous || !next) return 0;
  const offset = currentTime >= previous.timeline_start - 15 && currentTime <= previous.timeline_end + 15 ? currentTime - previous.timeline_start : -4;
  return Math.max(0, Math.min(newResult.duration - .1, next.timeline_start + offset));
}

async function setMode(mode, compare = true) {
  const previous = active();
  const wasPlaying = !audio.paused;
  const oldTime = audio.currentTime || 0;
  if (!state.results[mode]) {
    if (state.busy || !(state.results.automix || state.results.plain)) return;
    setBusy(true); $("error-banner").hidden = true;
    try { await createMix(mode, true); $("status-panel").hidden = true; }
    catch (error) { showError(error); return; }
    finally { setBusy(false); }
  }
  state.mode = mode;
  const result = active();
  const seekTo = compare ? positionForComparison(previous, result, oldTime) : 0;
  $("mode-automix").classList.toggle("selected", mode === "automix");
  $("mode-plain").classList.toggle("selected", mode === "plain");
  $("mode-automix").setAttribute("aria-pressed", String(mode === "automix"));
  $("mode-plain").setAttribute("aria-pressed", String(mode === "plain"));
  audio.src = `/api/mixes/${encodeURIComponent(result.id)}/audio`;
  audio.onloadedmetadata = () => { audio.currentTime = Math.min(seekTo, audio.duration || seekTo); updatePlayback(); if (wasPlaying && compare) playAudio(); };
  audio.load();
  $("waveform-empty").hidden = true;
  $("playhead").hidden = false;
  $("player-title").textContent = mode === "automix" ? "Your tracks, connected." : "The same tracks. A simple fade.";
  $("player-subtitle").textContent = `${result.tracks.length} tracks · ${result.transitions.length} transitions · ${mode === "automix" ? "Intelligent cue points" : "Plain equal-power crossfade"}`;
  $("mix-state").textContent = mode === "automix" ? "✦ AUTOMIX READY" : "PLAIN CROSSFADE";
  $("mix-state").classList.add("ready");
  $("duration").textContent = time(result.duration);
  $("audio-spec").textContent = `${((result.sample_rate || 44100) / 1000).toFixed(1)} kHz / STEREO`;
  $("seek").max = result.duration;
  ["play", "restart", "seek"].forEach(id => $(id).disabled = false);
  $("next-transition").disabled = !result.transitions.length;
  $("transition-select").disabled = !result.transitions.length;
  $("transition-select").innerHTML = result.transitions.map((transition, index) => `<option value="${index}">${String(index + 1).padStart(2, "0")} → ${String(index + 2).padStart(2, "0")}</option>`).join("");
  state.transition = Math.min(state.transition, Math.max(0, result.transitions.length - 1));
  $("transition-select").value = state.transition;
  $("audition-buttons").replaceChildren();
  result.transitions.forEach((transition, index) => { const button = document.createElement("button"); button.textContent = `↗ Hear transition ${String(index + 1).padStart(2, "0")}`; button.onclick = () => audition(index); $("audition-buttons").append(button); });
  $("transition-labels").innerHTML = result.transitions.map((transition, index) => `<div class="transition-label" style="left:${100 * transition.timeline_start / result.duration}%;width:${100 * (transition.timeline_end - transition.timeline_start) / result.duration}%"><span>T${index + 1}</span></div>`).join("");
  [["export-audio", "audio", "wav"], ["export-plan", "plan", "json"]].forEach(([id, route, extension]) => { $(id).href = `/api/mixes/${encodeURIComponent(result.id)}/${route}`; $(id).download = `automix-lab-${mode}.${extension}`; $(id).classList.remove("disabled"); $(id).setAttribute("aria-disabled", "false"); });
  renderQueue(); renderInspector(); drawAll(); updatePlayback();
}

function peaks(waveform) {
  if (!waveform) return [];
  const values = Array.isArray(waveform) ? waveform : waveform.peaks || waveform.values || [];
  return values.map(value => Array.isArray(value) ? Math.max(...value.map(Math.abs)) : typeof value === "object" ? Math.abs(value.peak ?? value.max ?? value.value ?? 0) : Math.abs(Number(value) || 0));
}

function drawWave(canvas, waveform, color, options = {}) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#37452c"; ctx.fillRect(0, height / 2, width, 1);
  const values = peaks(waveform);
  if (!values.length) return;
  const largest = Math.max(.01, ...values);
  const bars = Math.floor(width / 4);
  for (let index = 0; index < bars; index++) {
    const start = Math.floor(index * values.length / bars);
    const end = Math.max(start + 1, Math.floor((index + 1) * values.length / bars));
    let value = 0; for (let item = start; item < end; item++) value = Math.max(value, values[item] || 0);
    const barHeight = Math.max(2, value / largest * height * .8);
    const position = index / bars;
    ctx.fillStyle = options.multicolor ? colors[Math.min(colors.length - 1, getTrackAt(position * (active()?.duration || 1)))] : color;
    ctx.globalAlpha = options.highlight && (position < options.highlight[0] || position > options.highlight[1]) ? .27 : .85;
    ctx.fillRect(index * width / bars, (height - barHeight) / 2, 2, barHeight);
  }
  ctx.globalAlpha = 1;
  if (options.marker !== undefined) { ctx.fillStyle = "#e6f9b9"; ctx.fillRect(options.marker * width, 1, 1, height - 2); }
}

function getTrackAt(position) {
  let index = 0;
  for (const transition of active()?.transitions || []) if (position >= transition.timeline_start + transition.overlap_seconds / 2) index++;
  return index;
}

function drawAll() {
  const result = active();
  drawWave($("mix-waveform"), result?.waveform, "#dcf87a", { multicolor: true });
  const transition = result?.transitions[state.transition];
  const outgoing = result?.tracks[state.transition];
  const incoming = result?.tracks[state.transition + 1];
  drawWave($("out-wave"), outgoing?.waveform, colors[state.transition % colors.length], transition ? { highlight: [transition.outgoing_start / outgoing.duration, transition.outgoing_end / outgoing.duration], marker: transition.outgoing_start / outgoing.duration } : {});
  drawWave($("in-wave"), incoming?.waveform, colors[(state.transition + 1) % colors.length], transition ? { highlight: [transition.incoming_cue / incoming.duration, (transition.incoming_cue + transition.overlap_seconds) / incoming.duration], marker: transition.incoming_cue / incoming.duration } : {});
}

function renderInspector() {
  const transition = active()?.transitions[state.transition];
  if (!transition) return;
  $("out-title").textContent = transition.outgoing_title || active().tracks[state.transition]?.title;
  $("in-title").textContent = transition.incoming_title || active().tracks[state.transition + 1]?.title;
  $("tier").textContent = String(transition.tier || "crossfade").replace(/[_-]/g, " ").toUpperCase();
  $("cue-value").textContent = `${Number(transition.incoming_cue).toFixed(1)}s`;
  $("overlap-value").textContent = `${Number(transition.overlap_seconds).toFixed(1)}s`;
  const rate = ((Number(transition.incoming_playback_rate) || 1) - 1) * 100;
  $("tempo-value").textContent = `${rate > 0 ? "+" : ""}${rate.toFixed(1)}%`;
  const reasons = Array.isArray(transition.reasons) ? transition.reasons : [transition.reasons].filter(Boolean);
  $("transition-explanation").textContent = reasons.join(" ") || (state.mode === "plain" ? "A simple equal-power fade lets you compare this queue with the beat-aware AutoMix version." : `The incoming track starts at ${Number(transition.incoming_cue).toFixed(1)} seconds and blends in over ${Number(transition.overlap_seconds).toFixed(1)} seconds.`);
  const confidence = Math.round((Number(transition.confidence) || 0) * 100);
  const vocal = Math.round((Number(transition.vocal_overlap) || 0) * 100);
  const bassApplied = /beatmatched|dj.assisted/i.test(transition.tier);
  const vocalEvidence = transition.vocal_evidence === "model" ? "UMX-HQ model" : transition.vocal_evidence || "unavailable";
  const duck = Number(transition.vocal_duck_db) || 0;
  const skipped = transition.discarded_music_seconds == null ? "" : ` · ${Number(transition.discarded_music_seconds).toFixed(1)}s audible music skipped`;
  const exit = transition.mix_out_type === "full_track_end" ? "Plain fade at file end" : `${transition.mix_out_type === "sustained_outro_drop" ? "Outro" : "End-region"} exit at ${Number(transition.outgoing_end).toFixed(1)}s${skipped}`;
  const reverb = Number(transition.reverb_wet) ? `Stereo reverb ${Math.round(Number(transition.reverb_wet) * 100)}% · ${Number(transition.reverb_tail_seconds).toFixed(1)}s tail` : "No added reverb";
  $("inspector-detail").textContent = `${exit} · ${reverb} · Confidence ${confidence}% · Vocal overlap ${vocal}% (${vocalEvidence}) · Vocal duck ${duck ? `${duck.toFixed(1)} dB` : "not needed"} · Bass handoff ${bassApplied && transition.bass_handoff_seconds != null ? `${Number(transition.bass_handoff_seconds).toFixed(1)}s into fade` : "not applied"}`;
  drawAll();
}

function updatePlayback() {
  const result = active();
  const position = audio.currentTime || 0;
  $("current-time").textContent = time(position);
  $("seek").value = position;
  $("playhead").style.left = `${result ? Math.min(100, position / result.duration * 100) : 0}%`;
  const transition = result?.transitions[state.transition];
  const progress = transition ? Math.max(0, Math.min(1, (position - transition.timeline_start) / transition.overlap_seconds)) : 0;
  $("out-gain").style.width = `${Math.cos(progress * Math.PI / 2) * 100}%`;
  $("in-gain").style.width = `${Math.sin(progress * Math.PI / 2) * 100}%`;
  if (!audio.paused) requestAnimationFrame(updatePlayback);
}

async function playAudio() {
  try { await audio.play(); }
  catch (error) { showError(new Error(`Playback couldn't start: ${error.message}. Click play to retry.`)); }
}

function audition(index) {
  const transition = active()?.transitions[index];
  if (!transition) return;
  state.transition = index; $("transition-select").value = index;
  audio.currentTime = Math.max(0, transition.timeline_start - 4);
  renderInspector(); playAudio();
}

function showDialog(title, content) { $("dialog-title").textContent = title; $("dialog-body").innerHTML = content; $("info-dialog").showModal(); }

$("build-upload").onclick = () => buildSession();
["hero-upload", "queue-upload", "browse-files"].forEach(id => $(id).onclick = () => $("file-input").click());
$("file-input").onchange = event => chooseFiles(event.target.files);
$("reset-queue").onclick = () => { state.files = []; state.results = { automix: null, plain: null }; $("queue-actions").hidden = true; $("file-input").value = ""; clearPlayer(); renderQueue(); };
$("mode-automix").onclick = () => { if (state.mode !== "automix") setMode("automix"); };
$("mode-plain").onclick = () => { if (state.mode !== "plain") setMode("plain"); };
$("play").onclick = () => audio.paused ? playAudio() : audio.pause();
$("restart").onclick = () => { audio.currentTime = 0; updatePlayback(); };
$("next-transition").onclick = () => { const transitions = active()?.transitions || []; const next = transitions.findIndex(transition => transition.timeline_start > audio.currentTime + 5); audition(next < 0 ? 0 : next); };
$("seek").oninput = event => { audio.currentTime = Number(event.target.value); if (audio.paused) updatePlayback(); };
$("volume").oninput = event => { audio.volume = Number(event.target.value); };
audio.volume = .8;
audio.onplay = () => { $("play-symbol").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "Pause mix"); updatePlayback(); };
audio.onpause = () => { $("play-symbol").textContent = "▶"; $("play").setAttribute("aria-label", "Play mix"); };
audio.onended = () => { $("play-symbol").textContent = "▶"; $("play").setAttribute("aria-label", "Play mix"); updatePlayback(); };
audio.ontimeupdate = () => { if (audio.paused) updatePlayback(); };
audio.onerror = () => { if (audio.getAttribute("src")) showError("The mix audio could not be loaded. Rebuild the session if its cached file was removed."); };
$("waveform-wrap").onclick = event => { if (!active()) return; const rect = $("waveform-wrap").getBoundingClientRect(); audio.currentTime = Math.max(0, Math.min(active().duration, (event.clientX - rect.left) / rect.width * active().duration)); updatePlayback(); };
$("transition-select").onchange = event => { state.transition = Number(event.target.value); renderInspector(); updatePlayback(); };
$("dialog-close").onclick = () => $("info-dialog").close();
$("info-dialog").onclick = event => { if (event.target === $("info-dialog")) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } };
$("show-credits").onclick = () => showDialog("Your audio, your responsibility.", "<p>Process and distribute only recordings you have the right to use. Vocal activity uses the UMX-HQ source-separation model when available and an explicitly labelled DSP fallback otherwise.</p>");
for (const event of ["dragenter", "dragover"]) $("upload-zone").addEventListener(event, e => { e.preventDefault(); if (!state.busy) $("upload-zone").classList.add("dragover"); });
for (const event of ["dragleave", "drop"]) $("upload-zone").addEventListener(event, e => { e.preventDefault(); $("upload-zone").classList.remove("dragover"); });
$("upload-zone").addEventListener("drop", event => chooseFiles(event.dataTransfer.files));
new ResizeObserver(drawAll).observe($("waveform-wrap"));
document.addEventListener("keydown", event => { if (event.code === "Space" && !/INPUT|BUTTON|SELECT|TEXTAREA|A/.test(document.activeElement.tagName) && !$("info-dialog").open && active()) { event.preventDefault(); audio.paused ? playAudio() : audio.pause(); } });

async function init() {
  const loaded = await request("/api/latest").catch(error => { showError(error); return null; });
  if (loaded) {
    const latest = loaded;
    state.results.automix = latest.automix || null; state.results.plain = latest.plain || null;
    if (state.results.automix || state.results.plain) setMode(state.results.automix ? "automix" : "plain", false);
  }
  renderQueue();
  drawAll();
}
init();
