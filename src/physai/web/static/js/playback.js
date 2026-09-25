import * as net from "./net.js";

const SPEEDS = [0.5, 1, 2, 4];

let panel = null;
let episodeSelect = null;
let loadSection = null;
let activeSection = null;
let playButton = null;
let speedSelect = null;
let scrubber = null;
let statusEl = null;
let lastSaved = -1;
let scrubbing = false;
let active = false;
let playing = false;
let ready = false;

export function isActive() {
  return active;
}

async function refreshEpisodes() {
  const episodes = await fetch("/api/episodes").then((response) => response.json());
  episodeSelect.replaceChildren();
  episodes.forEach((episode) => {
    const option = document.createElement("option");
    option.value = episode.file;
    option.disabled = !episode.playable;
    const tag = episode.success ? "✓" : "✗";
    option.textContent = `${episode.file.replace(".npz", "")} ${tag} · ${episode.length} frames`;
    episodeSelect.appendChild(option);
  });
  // Newest first is what you usually want to review.
  if (episodeSelect.options.length) episodeSelect.selectedIndex = episodeSelect.options.length - 1;
}

function sendPlay(nextPlaying) {
  net.send({ type: "playback_play", playing: nextPlaying, speed: Number(speedSelect.value) });
}

export function init(panelEl) {
  panel = panelEl;
  episodeSelect = panel.querySelector("#playback-episode");
  loadSection = panel.querySelector("#playback-load-section");
  activeSection = panel.querySelector("#playback-active-section");
  playButton = panel.querySelector("#playback-play");
  speedSelect = panel.querySelector("#playback-speed");
  scrubber = panel.querySelector("#playback-scrubber");
  statusEl = panel.querySelector("#playback-status");
  SPEEDS.forEach((speed) => {
    const option = document.createElement("option");
    option.value = String(speed);
    option.textContent = `${speed}×`;
    if (speed === 1) option.selected = true;
    speedSelect.appendChild(option);
  });
  panel.querySelector("#playback-load").onclick = () => {
    if (episodeSelect.value) net.send({ type: "playback_load", file: episodeSelect.value });
  };
  panel.querySelector("#playback-exit").onclick = () => net.send({ type: "playback_exit" });
  panel.querySelector("#playback-prev").onclick = () =>
    net.send({ type: "playback_step", delta: -1 });
  panel.querySelector("#playback-next").onclick = () =>
    net.send({ type: "playback_step", delta: 1 });
  playButton.onclick = () => sendPlay(!playing);
  speedSelect.onchange = () => sendPlay(playing);
  scrubber.addEventListener("pointerdown", () => {
    scrubbing = true;
  });
  scrubber.addEventListener("pointerup", () => {
    scrubbing = false;
  });
  scrubber.addEventListener("input", () =>
    net.send({ type: "playback_seek", frame: Number(scrubber.value) }),
  );
  window.addEventListener("keydown", (event) => {
    if (!active || event.target instanceof HTMLSelectElement) return;
    if (event.key === "ArrowLeft") net.send({ type: "playback_step", delta: -1 });
    else if (event.key === "ArrowRight") net.send({ type: "playback_step", delta: 1 });
    else return;
    event.preventDefault();
  });
  ready = true;
}

export function applyState(state) {
  if (!ready) return;
  const playback = state.playback ?? { active: false };
  const enabled = !!state.recording?.enabled;
  panel.hidden = !enabled;
  active = enabled && playback.active;
  if (!enabled) return;
  // The episode list only changes when a take is saved.
  if (state.recording.episodes_saved !== lastSaved) {
    lastSaved = state.recording.episodes_saved;
    refreshEpisodes();
  }
  loadSection.hidden = active;
  activeSection.hidden = !active;
  // Loading needs a finished take, not one in progress.
  panel.querySelector("#playback-load").disabled = !!state.recording.active;
  if (!active) {
    playing = false;
    return;
  }
  playing = playback.playing;
  playButton.textContent = playing ? "Pause" : "Play";
  speedSelect.value = String(playback.speed);
  scrubber.max = String(playback.length - 1);
  if (!scrubbing) scrubber.value = String(playback.frame);
  const tag = playback.success === null ? "" : playback.success ? " ✓ success" : " ✗ fail";
  statusEl.textContent = `${playback.file.replace(".npz", "")}${tag} · frame ${playback.frame + 1}/${playback.length}`;
}
