import * as net from "./net.js";
import * as scene from "./scene.js";
import * as controls from "./controls.js";
import * as ui from "./ui.js";
import * as cameras from "./cameras.js";
import * as joints from "./joints.js";
import * as recording from "./recording.js";
import * as pose from "./pose.js";
import * as playback from "./playback.js";

const robotSelect = document.querySelector("#robot-select");
const robotInfo = new Map();
let activeRobot = "";

cameras.init(document.querySelector("#camera-grid"), document.querySelector("#camera-add"));
joints.init(document.querySelector("#joint-panel"));
recording.init(document.querySelector("#record-panel"));
pose.init(document.querySelector("#pose-rows"));
playback.init(document.querySelector("#playback-panel"));

const pauseButton = document.querySelector("#pause");
const resetButton = document.querySelector("#reset");
// Pause is world-atomic and server-owned: the button follows the reported
// state, so a second client or a reconnect never drifts out of sync.
let serverPaused = false;

net.on("open", () => {
  ui.setStatus("connected", "connected");
  net.send({ type: "select_robot", robot: activeRobot });
});
net.on("close", () => ui.setStatus("reconnecting", "reconnecting"));
net.on("error", (message) => ui.pushToast(message, "error"));
net.on("state", (state) => {
  scene.applyState(state);
  joints.applyState(state, activeRobot);
  recording.applyState(state);
  pose.applyState(state);
  playback.applyState(state);
  // Playback owns the paused world: live controls are off until it exits.
  const playingBack = playback.isActive();
  controls.setEnabled(!playingBack);
  resetButton.disabled = playingBack;
  pauseButton.disabled = playingBack;
  serverPaused = !!state.paused;
  pauseButton.textContent = serverPaused ? "Resume" : "Pause";
});

resetButton.onclick = () => {
  net.send({ type: "reset", robot: activeRobot });
};
pauseButton.onclick = () => {
  net.send({ type: "pause", robot: activeRobot, value: !serverPaused });
};

async function selectRobot(name) {
  activeRobot = name;
  controls.setActiveRobot(name);
  controls.configureControls(robotInfo.get(name));
  joints.configureForRobot(robotInfo.get(name));
  await scene.loadScene(name);
  cameras.setAvailableCameras(name, robotInfo.get(name)?.cameras || []);
  net.send({ type: "select_robot", robot: name });
}

async function initialize() {
  const robots = await fetch("/api/robots").then((response) => response.json());
  robots.forEach((robot) => {
    robotInfo.set(robot.name, robot);
    const option = document.createElement("option");
    option.value = robot.name;
    option.textContent = robot.name;
    robotSelect.appendChild(option);
  });
  const initial = robots[0]?.name || "";
  robotSelect.value = initial;
  await selectRobot(initial);
  robotSelect.onchange = () => selectRobot(robotSelect.value);
  net.connect();
}

initialize();
scene.render();
