const RAD_TO_DEG = 180 / Math.PI;
// How close to either end of a joint's range counts as "hit the limit",
// as a fraction of the full range -- percentage-based so it works the same
// for a wide-range joint and a narrow one like wrist_flex.
const NEAR_LIMIT_MARGIN = 0.04;

let rowsEl = null;
let contactSectionEl = null;
let contactDots = new Map();
let contactForces = new Map();
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
  contactForces = new Map(
    [...panel.querySelectorAll(".contact-row")].map((el) => [
      el.dataset.pad,
      el.querySelector("[data-force]"),
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
    const ratio = clamp01((position - min) / (max - min));
    entry.fill.style.width = `${(ratio * 100).toFixed(1)}%`;
    entry.value.textContent = `${(position * RAD_TO_DEG).toFixed(0)}°`;
    const nearLimit = ratio <= NEAR_LIMIT_MARGIN || ratio >= 1 - NEAR_LIMIT_MARGIN;
    entry.value.classList.toggle("is-near-limit", nearLimit);
    entry.fill.classList.toggle("is-near-limit", nearLimit);
  });

  if (contactSectionEl.hidden) return;
  // Normal force in newtons per pad, summed over everything that pad touches.
  const forces = new Map();
  (state.gripper_contacts || []).forEach((contact) => {
    if (contact.instance_id !== undefined && contact.instance_id !== activeRobot) return;
    forces.set(contact.pad, (forces.get(contact.pad) ?? 0) + (contact.force_n ?? 0));
  });
  contactDots.forEach((dot, pad) => dot.setAttribute("data-active", String(forces.has(pad))));
  // force_n is null while replaying an episode: the recorded qpos cannot
  // reproduce the actuator state behind the original squeeze.
  contactForces.forEach((el, pad) => {
    el.textContent = !forces.has(pad)
      ? "–"
      : state.playback?.active
        ? "n/a"
        : `${forces.get(pad).toFixed(2)} N`;
  });
}
