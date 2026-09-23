import * as net from "./net.js";

const controlHint = document.querySelector("#control-hint");
const controlButtons = new Map(
  [...document.querySelectorAll("[data-control-key]")].map((button) => [button.dataset.controlKey, button]),
);
const heldKeys = new Map(); // key -> hold-start timestamp (ms), for every ramped key (jog axes and gripper alike)
let gripper = 1;
let activeRobot = "";
let jogAxes = {};
let shoulderPanAxis = {};
let wristFlexAxis = {};
let wristRollAxis = {};
let gripperEnabled = false;

const JOG_RAMP_MS = 600; // ms of continuous hold to reach max speed
const JOG_LINEAR_MIN = 0.06; // m/s at a tap
const JOG_LINEAR_MAX = 0.18; // m/s ceiling after a sustained hold
const JOG_JOINT_MIN = 0.5; // rad/s at a tap (direct joint jog: shoulder pan/wrist)
const JOG_JOINT_MAX = 1.5; // rad/s ceiling after a sustained hold
// The gripper actuator runs uncapped force in the viewer (see run_sim.py) so
// grasps actually hold, which made the old fixed 0.04/tick step feel like a
// slam on a quick tap. Ramping it the same way as every other jog key keeps
// a tap gentle and only reaches full speed on a sustained hold.
const JOG_GRIPPER_MIN = 0.015; // aperture fraction per tick at a tap
const JOG_GRIPPER_MAX = 0.05; // aperture fraction per tick after a sustained hold

export function setActiveRobot(name) {
  activeRobot = name;
  gripper = 1;
  heldKeys.clear();
}

function isGripperKey(key) {
  return key === "f" || key === "g";
}

export function configureControls(robot) {
  jogAxes = { w: [1, 0, 0], s: [-1, 0, 0] };
  shoulderPanAxis = {};
  wristFlexAxis = {};
  wristRollAxis = {};
  gripperEnabled = false;
  if (robot?.capabilities?.includes("arm_kinematics")) {
    jogAxes.u = [0, 0, 1];
    jogAxes.j = [0, 0, -1];
    shoulderPanAxis = { a: -1, d: 1 };
    wristFlexAxis = { i: -1, k: 1 };
    wristRollAxis = { q: 1, e: -1 };
    gripperEnabled = robot.capabilities.includes("gripper");
    controlHint.textContent =
      "Jog: W/S x · U/J z · A/D shoulder pan · I/K wrist tilt · Q/E wrist roll · F/G gripper.";
  } else if (robot?.capabilities?.includes("base_velocity")) {
    controlHint.textContent = "Drive: W/S forward/reverse · A/D turn.";
  } else {
    controlHint.textContent = "Keyboard jog is not configured for this robot.";
  }
  controlButtons.forEach((button, key) => {
    const enabled =
      jogAxes[key] ||
      shoulderPanAxis[key] ||
      wristFlexAxis[key] ||
      wristRollAxis[key] ||
      (gripperEnabled && isGripperKey(key));
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
  const gripperKey = isGripperKey(key);
  if (gripperKey && !gripperEnabled) return;
  if (
    !jogAxes[key] &&
    !shoulderPanAxis[key] &&
    !wristFlexAxis[key] &&
    !wristRollAxis[key] &&
    !gripperKey
  )
    return;
  if (pressed) {
    if (!heldKeys.has(key)) heldKeys.set(key, performance.now());
  } else {
    heldKeys.delete(key);
  }
  setKeyVisual(key, pressed);
}

function rampedSpeed(holdStart, min, max) {
  const elapsed = performance.now() - holdStart;
  const rampFactor = Math.min(1, elapsed / JOG_RAMP_MS);
  return min + (max - min) * rampFactor;
}

function sendJog(force = false) {
  if (!net.isOpen()) return;
  if (!force && heldKeys.size === 0) return;
  const linear = [0, 0, 0];
  let shoulderPanRate = 0;
  let wristFlexRate = 0;
  let wristRollRate = 0;
  let gripperMoved = false;
  heldKeys.forEach((holdStart, key) => {
    const axis = jogAxes[key];
    if (axis) {
      const speed = rampedSpeed(holdStart, JOG_LINEAR_MIN, JOG_LINEAR_MAX);
      axis.forEach((value, index) => { linear[index] += value * speed; });
    }
    if (shoulderPanAxis[key]) {
      shoulderPanRate += shoulderPanAxis[key] * rampedSpeed(holdStart, JOG_JOINT_MIN, JOG_JOINT_MAX);
    }
    if (wristFlexAxis[key]) {
      wristFlexRate += wristFlexAxis[key] * rampedSpeed(holdStart, JOG_JOINT_MIN, JOG_JOINT_MAX);
    }
    if (wristRollAxis[key]) {
      wristRollRate += wristRollAxis[key] * rampedSpeed(holdStart, JOG_JOINT_MIN, JOG_JOINT_MAX);
    }
    if (gripperEnabled && isGripperKey(key)) {
      const step = rampedSpeed(holdStart, JOG_GRIPPER_MIN, JOG_GRIPPER_MAX);
      gripper = key === "f" ? Math.min(1, gripper + step) : Math.max(0, gripper - step);
      gripperMoved = true;
    }
  });
  const action = {
    mode: "twist", linear: { x: linear[0], y: linear[1], z: linear[2] },
    angular: { x: shoulderPanRate, y: wristFlexRate, z: wristRollRate },
  };
  if (gripperMoved) action.gripper = gripper;
  net.send({ type: "command", robot: activeRobot, action });
}

function isRampedKey(key) {
  return jogAxes[key] || shoulderPanAxis[key] || wristFlexAxis[key] || wristRollAxis[key] || isGripperKey(key);
}

window.addEventListener("keydown", (event) => {
  const key = event.key.toLowerCase();
  if (isRampedKey(key)) { setHeldKey(key, true); event.preventDefault(); sendJog(); }
});
window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  setHeldKey(key, false);
  if (isRampedKey(key)) {
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
    sendJog();
  });
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("lostpointercapture", release);
});
window.addEventListener("blur", () => {
  [...heldKeys.keys()].forEach((key) => setHeldKey(key, false));
  sendJog(true);
});
setInterval(sendJog, 40);
