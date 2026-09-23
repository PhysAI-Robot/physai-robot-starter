const RAD_TO_DEG = 180 / Math.PI;

let rowsEl = null;
let contactSectionEl = null;
let contactDots = new Map();
let jointNames = [];
let jointLimits = {};
let rows = new Map();

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function buildRow(name) {
  const row = document.createElement("div");
  row.className = "joint-row";
  const label = document.createElement("span");
  label.className = "joint-name";
  label.textContent = name;
  label.title = name;
  const bar = document.createElement("div");
  bar.className = "joint-bar";
  const fill = document.createElement("div");
  fill.className = "joint-bar-fill";
  bar.appendChild(fill);
  const value = document.createElement("span");
  value.className = "joint-value";
  value.textContent = "–";
  row.appendChild(label);
  row.appendChild(bar);
  row.appendChild(value);
  return { row, fill, value };
}

export function init(panel) {
  rowsEl = panel.querySelector("#joint-rows");
  contactSectionEl = panel.querySelector("#contact-rows");
  contactDots = new Map(
    [...panel.querySelectorAll(".contact-row")].map((el) => [
      el.dataset.pad,
      el.querySelector(".contact-dot"),
    ]),
  );
}

export function configureForRobot(robot) {
  jointNames = robot?.joint_names || [];
  jointLimits = robot?.joint_limits || {};
  rowsEl.replaceChildren();
  rows = new Map();
  jointNames.forEach((name) => {
    const entry = buildRow(name);
    rows.set(name, entry);
    rowsEl.appendChild(entry.row);
  });
  contactSectionEl.hidden = !(robot?.capabilities || []).includes("gripper");
  contactDots.forEach((dot) => dot.setAttribute("data-active", "false"));
}

export function applyState(state, activeRobot) {
  if (jointNames.length === 0) return;
  const known = new Set(jointNames);
  state.joints.forEach((joint) => {
    if (!known.has(joint.name)) return;
    if (joint.instance_id !== undefined && joint.instance_id !== activeRobot) return;
    const entry = rows.get(joint.name);
    if (!entry) return;
    const position = joint.qpos[0];
    const [min, max] = jointLimits[joint.name] || [-Math.PI, Math.PI];
    entry.fill.style.width = `${(clamp01((position - min) / (max - min)) * 100).toFixed(1)}%`;
    entry.value.textContent = `${(position * RAD_TO_DEG).toFixed(0)}°`;
  });

  if (contactSectionEl.hidden) return;
  const active = new Set();
  (state.gripper_contacts || []).forEach((contact) => {
    if (contact.instance_id !== undefined && contact.instance_id !== activeRobot) return;
    active.add(contact.pad);
  });
  contactDots.forEach((dot, pad) => dot.setAttribute("data-active", String(active.has(pad))));
}
