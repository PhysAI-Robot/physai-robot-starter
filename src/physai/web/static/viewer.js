import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const viewport = document.querySelector("#viewport");
const status = document.querySelector("#status");
const telemetry = document.querySelector("#telemetry");
const controlHint = document.querySelector("#control-hint");
const robotSelect = document.querySelector("#robot-select");
const controlButtons = new Map(
  [...document.querySelectorAll("[data-control-key]")].map((button) => [button.dataset.controlKey, button]),
);
const heldKeys = new Set();
const heldGripperKeys = new Set();
let gripper = 1;
let activeRobot = "";
const robotInfo = new Map();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdfe6e2);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.up.set(0, 0, 1);
camera.position.set(0.7, -0.9, 0.55);
const controls = new OrbitControls(camera, viewport);
controls.target.set(0, 0, 0.2);
controls.enableDamping = true;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0xaebcb4, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(2.5, -3, 4);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xd7e7ff, 1.1);
fillLight.position.set(-3, 1, 2.5);
scene.add(fillLight);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(6, 6),
  new THREE.MeshStandardMaterial({ color: 0xcbd4cf, roughness: 0.9 }),
);
floor.position.z = -0.01;
scene.add(floor);
const grid = new THREE.GridHelper(6, 30, 0x71847b, 0xa8b5ae);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.002;
scene.add(grid);
const meshes = new Map();
const targetTransforms = new Map();
const geometryTypes = {
  box: THREE.BoxGeometry,
  sphere: THREE.SphereGeometry,
  cylinder: THREE.CylinderGeometry,
};

function resize() {
  const rect = viewport.getBoundingClientRect();
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}
window.addEventListener("resize", resize);
resize();

function addPrimitive(item) {
  const Constructor = geometryTypes[item.type] || THREE.BoxGeometry;
  const size = item.size;
  const isTargetPad = item.name === "target_pad";
  const geometry = isTargetPad
    ? new THREE.CircleGeometry(size[0], 32)
    : item.type === "sphere"
    ? new Constructor(size[0], 20, 12)
    : item.type === "cylinder"
      ? new Constructor(size[0], size[0], size[1] * 2, 16)
      : new Constructor(size[0] * 2, size[1] * 2, size[2] * 2);
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(item.rgba[0], item.rgba[1], item.rgba[2]),
    roughness: 0.78,
    transparent: item.rgba[3] < 1,
    opacity: item.rgba[3],
  });
  const group = new THREE.Group();
  const mesh = new THREE.Mesh(geometry, material);
  if (item.type === "cylinder" && !isTargetPad) mesh.rotation.x = Math.PI / 2;
  group.add(mesh);
  scene.add(group);
  meshes.set(item.id, group);
}

async function addGeometry(item) {
  if (item.type !== "mesh" || !item.visual || item.mesh_id === null) {
    if (item.type !== "mesh" || item.visual) addPrimitive(item);
    return;
  }
  try {
    const buffer = await fetch(`/api/mesh/${item.mesh_id}?robot=${encodeURIComponent(activeRobot)}`).then((response) => response.arrayBuffer());
    const header = new DataView(buffer, 0, 8);
    const vertexCount = header.getUint32(0, true);
    const indexCount = header.getUint32(4, true);
    const vertexBytes = vertexCount * 3 * 4;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(
      new Float32Array(buffer, 8, vertexCount * 3), 3,
    ));
    geometry.setIndex(new THREE.BufferAttribute(
      new Uint32Array(buffer, 8 + vertexBytes, indexCount), 1,
    ));
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(item.rgba[0], item.rgba[1], item.rgba[2]),
      roughness: 0.72,
      metalness: 0.08,
    });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    meshes.set(item.id, mesh);
  } catch (_error) {
    addPrimitive(item);
  }
}

async function loadScene() {
  meshes.forEach((mesh) => scene.remove(mesh));
  meshes.clear();
  targetTransforms.clear();
  const manifest = await fetch(`/api/scene?robot=${encodeURIComponent(activeRobot)}`).then((response) => response.json());
  await Promise.all(manifest.geometries.map(addGeometry));
  status.textContent = `${manifest.robot} · live`;
}

function applyState(state) {
  state.geometries.forEach((item) => {
    const mesh = meshes.get(item.id);
    if (!mesh) return;
    const target = targetTransforms.get(item.id) || {
      position: new THREE.Vector3(),
      quaternion: new THREE.Quaternion(),
    };
    target.position.fromArray(item.position);
    target.quaternion.fromArray(item.quaternion);
    targetTransforms.set(item.id, target);
  });
  telemetry.textContent = `step ${state.step} · sim ${state.sim_time.toFixed(2)}s · ${state.geometries.length} geometries`;
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "error") {
      status.textContent = message.message;
      return;
    }
    applyState(message);
  };
  socket.onopen = () => {
    status.textContent = "connected";
    socket.send(JSON.stringify({ type: "select_robot", robot: activeRobot }));
  };
  socket.onclose = () => {
    status.textContent = "reconnecting";
    setTimeout(connect, 1000);
  };
  window.viewerSocket = socket;
}
window.addEventListener("beforeunload", () => {
  if (window.viewerSocket?.readyState === WebSocket.OPEN) {
    window.viewerSocket.send(JSON.stringify({ type: "release_control" }));
  }
});

document.querySelector("#reset").onclick = () => {
  window.viewerSocket?.send(JSON.stringify({ type: "reset", robot: activeRobot }));
};
document.querySelector("#pause").onclick = (event) => {
  const paused = event.target.textContent === "Pause";
  event.target.textContent = paused ? "Resume" : "Pause";
  window.viewerSocket?.send(JSON.stringify({ type: "pause", robot: activeRobot, value: paused }));
};

let jogAxes = {};
let yawAxes = {};
let tiltAxes = {};
let gripperEnabled = false;
function configureControls(robot) {
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
  if (!window.viewerSocket || window.viewerSocket.readyState !== WebSocket.OPEN) return;
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
  window.viewerSocket?.send(JSON.stringify({ type: "command", robot: activeRobot, action }));
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
function refreshCameras() {
  const cameras = new Set(robotInfo.get(activeRobot)?.cameras || []);
  document.querySelectorAll("[data-camera]").forEach((image) => {
    const name = image.dataset.camera;
    image.closest("figure").hidden = !cameras.has(name);
    if (!cameras.has(name)) return;
    image.src = `/api/camera/${name}.jpg?robot=${encodeURIComponent(activeRobot)}&t=${Date.now()}`;
  });
}
document.querySelectorAll("[data-camera]").forEach((image) => {
  const name = image.dataset.camera;
  setInterval(() => {
    if (image.closest("figure").hidden) return;
    image.src = `/api/camera/${name}.jpg?robot=${encodeURIComponent(activeRobot)}&t=${Date.now()}`;
  }, 250);
});

function render() {
  meshes.forEach((mesh, id) => {
    const target = targetTransforms.get(id);
    if (!target) return;
    mesh.position.lerp(target.position, 0.7);
    mesh.quaternion.slerp(target.quaternion, 0.7);
  });
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
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
  activeRobot = robots[0]?.name || "";
  robotSelect.value = activeRobot;
  configureControls(robotInfo.get(activeRobot));
  robotSelect.onchange = async () => {
    activeRobot = robotSelect.value;
    gripper = 1;
    heldKeys.clear();
    heldGripperKeys.clear();
    configureControls(robotInfo.get(activeRobot));
    await loadScene();
    refreshCameras();
    window.viewerSocket?.send(JSON.stringify({ type: "select_robot", robot: activeRobot }));
  };
  await loadScene();
  refreshCameras();
  connect();
}
initialize();
render();