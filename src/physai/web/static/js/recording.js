import * as net from "./net.js";
import * as ui from "./ui.js";

let panel = null;
let startButton = null;
let saveButtons = null;
let statusEl = null;
let lastError = null;
let ready = false;

function send(type, extra = {}) {
  net.send({ type, ...extra });
}

export function init(panelEl) {
  panel = panelEl;
  startButton = panel.querySelector("#record-start");
  saveButtons = [...panel.querySelectorAll("[data-record-stop]")];
  statusEl = panel.querySelector("#record-status");
  startButton.onclick = () => send("record_start");
  panel.querySelector("#record-success").onclick = () => send("record_stop", { success: true });
  panel.querySelector("#record-fail").onclick = () => send("record_stop", { success: false });
  panel.querySelector("#record-discard").onclick = () => send("record_stop", { success: null });
  ready = true;
}

export function applyState(state) {
  if (!ready) return;
  const recording = state.recording;
  panel.hidden = !recording?.enabled;
  if (panel.hidden) return;
  startButton.hidden = recording.active;
  startButton.disabled = !!state.playback?.active;
  saveButtons.forEach((button) => {
    button.hidden = !recording.active;
  });
  const saved = `${recording.episodes_saved} saved (${recording.successes} success)`;
  const waiting = recording.waiting_for?.length
    ? ` · waiting for camera ${recording.waiting_for.join(", ")}`
    : "";
  statusEl.textContent = recording.active
    ? `recording · ${recording.frames} frames${waiting} · ${saved}`
    : saved;
  if (recording.error && recording.error !== lastError) {
    ui.pushToast(recording.error, "error");
  }
  lastError = recording.error;
}
