import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as ui from "./ui.js";

const viewport = document.querySelector("#viewport");
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

let activeRobot = "";

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

export async function loadScene(robotName) {
  activeRobot = robotName;
  meshes.forEach((mesh) => scene.remove(mesh));
  meshes.clear();
  targetTransforms.clear();
  ui.showLoading();
  try {
    const manifest = await fetch(`/api/scene?robot=${encodeURIComponent(activeRobot)}`).then((response) => response.json());
    await Promise.all(manifest.geometries.map(addGeometry));
    return manifest;
  } finally {
    ui.hideLoading();
  }
}

export function applyState(state) {
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
  ui.setTelemetry(`step ${state.step} · sim ${state.sim_time.toFixed(2)}s · ${state.geometries.length} geometries`);
}

export function render() {
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
