import * as net from "./net.js";

const controlHint = document.querySelector("#control-hint");
const controlButtons = new Map(
  [...document.querySelectorAll("[data-control-key]")].map((button) => [button.dataset.controlKey, button]),
);
const heldKeys = new Set();
const heldGripperKeys = new Set();
let gripper = 1;
let activeRobot = "";
let jogAxes = {};
let yawAxes = {};
let tiltAxes = {};
let gripperEnabled = false;

export function setActiveRobot(name) {
  activeRobot = name;
  gripper = 1;
  heldKeys.clear();
  heldGripperKeys.clear();
}

export function configureControls(robot) {
  jogAxes = { w: [1, 0, 0], s: [-1, 0, 0] };
  yawAxes = { a: 1, d: -1 };
  tiltAxes = {};
  gripperEnabled = false;
  if (robot?.capabilities?.includes("arm_kinematics")) {
    jogAxes.q = [0, 0, 1];
    jogAxes.e = [0, 0, -1];
    tiltAxes = { r: 1, f: -1 };
    gripperEnabled = robot.capabilities.includes("gripper");
    controlHint.textContent = "Jog: W/S x · A/D base yaw · Q/E z · R/F grip tilt · O/C gripper.";
  } else if (robot?.capabilities?.includes("base_velocity")) {
    controlHint.textContent = "Drive: W/S forward/reverse · A/D turn.";
  } else {
    controlHint.textContent = "Keyboard jog is not configured for this robot.";
  }
  controlButtons.forEach((button, key) => {
    const enabled = jogAxes[key] || yawAxes[key] || tiltAxes[key] || (gripperEnabled && (key === "o" || key === "c"));
    button.hidden = !enabled;
    button.classList.remove("is-pressed");
    button.setAttribute("aria-disabled", String(!enabled));
  });
}

function setKeyVisual(key, pressed) {
  const button = controlButtons.get(key);
  if (!button || button.hidden) return;
  button.classList.toggle("is-pressed", pressed);
  button.setAttribute("aria-pressed", String(pressed));
}

function setHeldKey(key, pressed) {
  const isGripperKey = key === "o" || key === "c";
  if (isGripperKey && !gripperEnabled) return;
  if (!jogAxes[key] && !yawAxes[key] && !tiltAxes[key] && !isGripperKey) return;
  const heldSet = isGripperKey ? heldGripperKeys : heldKeys;
  if (pressed) heldSet.add(key);
  else heldSet.delete(key);
  setKeyVisual(key, pressed);
}

function sendJog(force = false) {
  if (!net.isOpen()) return;
  if (!force && heldKeys.size === 0 && heldGripperKeys.size === 0) return;
  const linear = [0, 0, 0];
  let yaw = 0;
  let tilt = 0;
  if (gripperEnabled && heldGripperKeys.has("o")) gripper = Math.min(1, gripper + 0.04);
  if (gripperEnabled && heldGripperKeys.has("c")) gripper = Math.max(0, gripper - 0.04);
  heldKeys.forEach((key) => {
    const axis = jogAxes[key];
    if (axis) axis.forEach((value, index) => { linear[index] += value * 0.06; });
    yaw += yawAxes[key] || 0;
    tilt += tiltAxes[key] || 0;
  });
  const action = {
    mode: "twist", linear: { x: linear[0], y: linear[1], z: linear[2] },
    angular: { x: 0, y: tilt * 0.5, z: yaw * 0.8 },
  };
  if (gripperEnabled && heldGripperKeys.size > 0) action.gripper = gripper;
  net.send({ type: "command", robot: activeRobot, action });
}

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (jogAxes[key] || yawAxes[key] || tiltAxes[key]) { setHeldKey(key, true); event.preventDefault(); sendJog(); }
  if (gripperEnabled && (key === "o" || key === "c")) { setHeldKey(key, true); event.preventDefault(); sendJog(true); }
});
window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  setHeldKey(key, false);
  if (jogAxes[key] || yawAxes[key] || tiltAxes[key]) {
    if (heldKeys.size === 0) sendJog(true);
    else sendJog();
  }
});
controlButtons.forEach((button, key) => {
  const release = () => {
    setHeldKey(key, false);
    sendJog(true);
  };
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    button.setPointerCapture(event.pointerId);
    setHeldKey(key, true);
    sendJog(key === "o" || key === "c");
  });
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("lostpointercapture", release);
});
window.addEventListener("blur", () => {
  [...heldKeys, ...heldGripperKeys].forEach((key) => setHeldKey(key, false));
  sendJog(true);
});
setInterval(sendJog, 40);
