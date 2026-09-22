import { startCameraStream } from "./ui.js";

const STORAGE_PREFIX = "physai-camera-slots:";
const DEFAULT_SLOT_COUNT = 2;

let gridEl = null;
let addButtonEl = null;
let activeRobot = "";
let availableCameras = [];
let slots = [];

function loadSavedNames(robotName) {
  try {
    const raw = localStorage.getItem(`${STORAGE_PREFIX}${robotName}`);
    const parsed = raw ? JSON.parse(raw) : null;
    return Array.isArray(parsed) ? parsed : null;
  } catch (_error) {
    // Storage may be unavailable (private browsing); fall back to defaults.
    return null;
  }
}

function persist() {
  try {
    const names = slots.map((slot) => slot.select.value);
    localStorage.setItem(`${STORAGE_PREFIX}${activeRobot}`, JSON.stringify(names));
  } catch (_error) {
    // Storage may be unavailable (private browsing); layout just won't persist.
  }
}

function updateAddButtonState() {
  if (!addButtonEl) return;
  addButtonEl.disabled = slots.length >= availableCameras.length;
}

function updateRemoveButtonsState() {
  const disabled = slots.length <= 1;
  slots.forEach((slot) => {
    slot.removeButton.disabled = disabled;
  });
}

function nextDefaultName() {
  const used = new Set(slots.map((slot) => slot.select.value));
  return availableCameras.find((name) => !used.has(name)) || availableCameras[0];
}

function removeSlot(slot) {
  if (slots.length <= 1) return;
  slots = slots.filter((entry) => entry !== slot);
  slot.figure.remove();
  updateAddButtonState();
  updateRemoveButtonsState();
  persist();
}

function createSlot(name) {
  const figure = document.createElement("figure");
  const figcaption = document.createElement("figcaption");
  const select = document.createElement("select");
  select.setAttribute("aria-label", "Camera feed");
  availableCameras.forEach((cameraName) => {
    const option = document.createElement("option");
    option.value = cameraName;
    option.textContent = cameraName;
    select.appendChild(option);
  });
  select.value = name;
  const removeButton = document.createElement("button");
  removeButton.type = "button";
  removeButton.className = "camera-remove";
  removeButton.textContent = "×";
  removeButton.setAttribute("aria-label", "Remove camera panel");
  figcaption.appendChild(select);
  figcaption.appendChild(removeButton);
  const img = document.createElement("img");
  img.dataset.camera = name;
  img.alt = `${name} camera feed`;
  figure.appendChild(figcaption);
  figure.appendChild(img);

  const slot = { figure, select, img, removeButton };
  select.onchange = () => {
    img.dataset.camera = select.value;
    img.alt = `${select.value} camera feed`;
    startCameraStream(img, select.value, activeRobot);
    persist();
  };
  removeButton.onclick = () => removeSlot(slot);
  return slot;
}

function addSlot(name) {
  const slot = createSlot(name);
  slots.push(slot);
  gridEl.appendChild(slot.figure);
  startCameraStream(slot.img, name, activeRobot);
  updateAddButtonState();
  updateRemoveButtonsState();
  return slot;
}

export function init(grid, addButton) {
  gridEl = grid;
  addButtonEl = addButton;
  addButtonEl.onclick = () => {
    if (availableCameras.length === 0 || slots.length >= availableCameras.length) return;
    addSlot(nextDefaultName());
    persist();
  };
}

export function setAvailableCameras(robotName, cameraNames) {
  activeRobot = robotName;
  availableCameras = cameraNames;
  gridEl.replaceChildren();
  slots = [];
  if (availableCameras.length === 0) {
    updateAddButtonState();
    return;
  }
  const saved = loadSavedNames(robotName);
  const names = (saved || []).filter((name) => availableCameras.includes(name));
  const initial = names.length > 0 ? names : availableCameras.slice(0, DEFAULT_SLOT_COUNT);
  initial.forEach((name) => addSlot(name));
  updateAddButtonState();
  updateRemoveButtonsState();
}
