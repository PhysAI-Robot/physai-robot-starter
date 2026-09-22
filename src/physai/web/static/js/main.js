import * as net from "./net.js";
import * as scene from "./scene.js";
import * as controls from "./controls.js";
import * as ui from "./ui.js";
import * as cameras from "./cameras.js";

const robotSelect = document.querySelector("#robot-select");
const robotInfo = new Map();
let activeRobot = "";

cameras.init(document.querySelector("#camera-grid"), document.querySelector("#camera-add"));

net.on("open", () => {
  ui.setStatus("connected", "connected");
  net.send({ type: "select_robot", robot: activeRobot });
});
net.on("close", () => ui.setStatus("reconnecting", "reconnecting"));
net.on("error", (message) => ui.pushToast(message, "error"));
net.on("state", (state) => scene.applyState(state));

document.querySelector("#reset").onclick = () => {
  net.send({ type: "reset", robot: activeRobot });
};
document.querySelector("#pause").onclick = (event) => {
  const paused = event.target.textContent === "Pause";
  event.target.textContent = paused ? "Resume" : "Pause";
  net.send({ type: "pause", robot: activeRobot, value: paused });
};

async function selectRobot(name) {
  activeRobot = name;
  controls.setActiveRobot(name);
  controls.configureControls(robotInfo.get(name));
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
