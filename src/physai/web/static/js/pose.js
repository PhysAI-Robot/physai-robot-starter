const RAD_TO_DEG = 180 / Math.PI;
const AXES = ["x", "y", "z", "roll", "pitch", "yaw"];

// Beyond this |pitch| roll and yaw are ill-conditioned: a tiny orientation
// change swings them by roughly 1 / cos(pitch) (5.8x at 80 deg).
const GIMBAL_LOCK_PITCH_DEG = 80;
const REFERENCE_LABEL = { tool: "tip", ee_pose: "end-effector" };

let sectionEl = null;
let frameEl = null;
let lockEl = null;
let valueEls = new Map();

// Intrinsic ZYX (yaw-pitch-roll) angles in degrees from an xyzw quaternion.
// Near pitch = +/-90 deg roll and yaw are not independent (gimbal lock), so
// they swing wildly while the orientation itself is fine.
export function quatToRpyDeg([x, y, z, w]) {
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  const pitch = Math.asin(Math.min(1, Math.max(-1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [roll * RAD_TO_DEG, pitch * RAD_TO_DEG, yaw * RAD_TO_DEG];
}

export function isNearGimbalLock(pitchDeg) {
  return Math.abs(pitchDeg) > GIMBAL_LOCK_PITCH_DEG;
}

export function init(section) {
  sectionEl = section;
  frameEl = section.querySelector("#pose-frame");
  lockEl = section.querySelector("#pose-lock");
  valueEls = new Map(AXES.map((axis) => [axis, section.querySelector(`[data-pose="${axis}"]`)]));
}

export function applyState(state) {
  if (!sectionEl) return;
  const pose = state.ee_pose;
  sectionEl.hidden = !pose;
  if (!pose) return;
  frameEl.textContent = `(${REFERENCE_LABEL[pose.reference] ?? pose.reference}, ${pose.frame_id})`;
  const [x, y, z] = pose.position.map((meters) => (meters * 1000).toFixed(1));
  const angles = quatToRpyDeg(pose.orientation_xyzw);
  const [roll, pitch, yaw] = angles.map((deg) => deg.toFixed(1));
  const text = { x, y, z, roll, pitch, yaw };
  AXES.forEach((axis) => {
    valueEls.get(axis).textContent = text[axis];
  });
  const locked = isNearGimbalLock(angles[1]);
  ["roll", "yaw"].forEach((axis) => {
    valueEls.get(axis).classList.toggle("is-near-lock", locked);
  });
  lockEl.hidden = !locked;
}
