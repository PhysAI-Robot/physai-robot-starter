const statusDot = document.querySelector("#status-dot");
const statusText = document.querySelector("#status");
const telemetry = document.querySelector("#telemetry");
const sceneLoading = document.querySelector("#scene-loading");
const toastRegion = document.querySelector("#toast-region");
const themeToggle = document.querySelector("#theme-toggle");
const sunIcon = themeToggle.querySelector(".icon-sun");
const moonIcon = themeToggle.querySelector(".icon-moon");
const darkMediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

export function setStatus(state, text) {
  statusDot.dataset.state = state;
  statusText.textContent = text ?? state;
}

export function setTelemetry(text) {
  telemetry.textContent = text;
}

export function showLoading() {
  sceneLoading.hidden = false;
}

export function hideLoading() {
  sceneLoading.hidden = true;
}

const recentToasts = new Map();
const TOAST_DEDUPE_MS = 4000;

export function pushToast(message, level = "info") {
  const now = Date.now();
  const last = recentToasts.get(message);
  if (last && now - last < TOAST_DEDUPE_MS) return;
  recentToasts.set(message, now);
  const toast = document.createElement("div");
  toast.className = level === "error" ? "toast toast-error" : "toast";
  toast.textContent = message;
  toastRegion.appendChild(toast);
  setTimeout(() => toast.remove(), TOAST_DEDUPE_MS);
}

function resolvedTheme() {
  return document.documentElement.dataset.theme || (darkMediaQuery.matches ? "dark" : "light");
}

function syncThemeIcon() {
  const isDark = resolvedTheme() === "dark";
  sunIcon.hidden = !isDark;
  moonIcon.hidden = isDark;
  themeToggle.setAttribute("aria-pressed", String(isDark));
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("physai-theme", theme);
  } catch (_error) {
    // Storage may be unavailable (private browsing); theme still applies for this load.
  }
  syncThemeIcon();
}

themeToggle.addEventListener("click", () => {
  applyTheme(resolvedTheme() === "dark" ? "light" : "dark");
});
darkMediaQuery.addEventListener("change", () => {
  // Only follow the OS live if the viewer has no explicit manual override.
  if (!document.documentElement.dataset.theme) syncThemeIcon();
});
syncThemeIcon();

function startCameraStream(image, name, robotName) {
  image.onerror = null;
  image.src = `/api/camera/${name}/stream?robot=${encodeURIComponent(robotName)}`;
  image.onerror = () => {
    if (image.closest("figure").hidden) return;
    setTimeout(() => startCameraStream(image, name, robotName), 1500);
  };
}

export function updateCameras(robotName, cameras) {
  const visible = new Set(cameras);
  document.querySelectorAll("[data-camera]").forEach((image) => {
    const name = image.dataset.camera;
    const isVisible = visible.has(name);
    image.closest("figure").hidden = !isVisible;
    image.onerror = null;
    if (!isVisible) {
      image.removeAttribute("src");
      return;
    }
    startCameraStream(image, name, robotName);
  });
}
