const dial = document.getElementById("dial");
const dialFace = dial.querySelector(".dial-face");
const readout = document.getElementById("orientationReadout");
const volumeKnob = document.getElementById("volumeKnob");
const volumeFace = volumeKnob.querySelector(".volume-face");

const orientations = ["Repair", "Automation", "Habitation", "Biodiversity"];

let angle = 28;
let dragging = false;

let volumeValue = 0.72;
let volumeDragging = false;
let lastVolumePointerAngle = null;

const VOLUME_MIN_ANGLE = -135;
const VOLUME_MAX_ANGLE = 135;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function normaliseAngle(value) {
  return ((value % 360) + 360) % 360;
}

function normaliseSignedAngle(value) {
  return ((value + 180) % 360 + 360) % 360 - 180;
}

function shortestAngularDelta(current, previous) {
  let delta = current - previous;

  if (delta > 180) delta -= 360;
  if (delta < -180) delta += 360;

  return delta;
}

function orientationBlend(degrees) {
  const scaled = normaliseAngle(degrees) / 90;
  const leftIndex = Math.floor(scaled) % orientations.length;
  const fraction = scaled - Math.floor(scaled);
  const rightIndex = (leftIndex + 1) % orientations.length;

  return {
    left: orientations[leftIndex],
    right: orientations[rightIndex],
    leftWeight: 1 - fraction,
    rightWeight: fraction
  };
}

function updateDial(nextAngle) {
  angle = normaliseAngle(nextAngle);
  dialFace.style.setProperty("--dial-angle", `${angle}deg`);
  dial.setAttribute("aria-valuenow", String(Math.round(angle)));

  const blend = orientationBlend(angle);
  const leftPercent = Math.round(blend.leftWeight * 100);
  const rightPercent = Math.round(blend.rightWeight * 100);

  readout.textContent =
    rightPercent === 0
      ? `${blend.left} 100%`
      : `${blend.left} ${leftPercent}% · ${blend.right} ${rightPercent}%`;

  window.dispatchEvent(new CustomEvent("listeningforward:tune", {
    detail: {
      angle,
      orientations: {
        [blend.left.toLowerCase()]: blend.leftWeight,
        [blend.right.toLowerCase()]: blend.rightWeight
      }
    }
  }));
}

function pointerAngle(event, element) {
  const rect = element.getBoundingClientRect();
  const centreX = rect.left + rect.width / 2;
  const centreY = rect.top + rect.height / 2;
  const deltaX = event.clientX - centreX;
  const deltaY = event.clientY - centreY;

  return Math.atan2(deltaY, deltaX) * (180 / Math.PI) + 90;
}

/* Main tuning dial: full circular motion. */
dial.addEventListener("pointerdown", event => {
  window.dispatchEvent(new CustomEvent("listeningforward:start"));
  dragging = true;
  dial.setPointerCapture(event.pointerId);
  updateDial(pointerAngle(event, dial));
});

dial.addEventListener("pointermove", event => {
  if (!dragging) return;
  updateDial(pointerAngle(event, dial));
});

dial.addEventListener("pointerup", event => {
  dragging = false;

  if (dial.hasPointerCapture(event.pointerId)) {
    dial.releasePointerCapture(event.pointerId);
  }
});

dial.addEventListener("pointercancel", () => {
  dragging = false;
});

dial.addEventListener("keydown", event => {
  window.dispatchEvent(new CustomEvent("listeningforward:start"));

  const fineStep = event.shiftKey ? 1 : 4;

  if (event.key === "ArrowRight" || event.key === "ArrowUp") {
    event.preventDefault();
    updateDial(angle + fineStep);
  }

  if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
    event.preventDefault();
    updateDial(angle - fineStep);
  }

  if (event.key === "Home") {
    event.preventDefault();
    updateDial(0);
  }
});

/* Volume: bounded physical sweep with genuine MIN and MAX stops. */
function volumeAngleFromValue(value) {
  return VOLUME_MIN_ANGLE
    + clamp(value, 0, 1) * (VOLUME_MAX_ANGLE - VOLUME_MIN_ANGLE);
}

function volumeValueFromAngle(value) {
  return (value - VOLUME_MIN_ANGLE)
    / (VOLUME_MAX_ANGLE - VOLUME_MIN_ANGLE);
}

function updateVolume(nextValue, emitEvent = true) {
  volumeValue = clamp(nextValue, 0, 1);

  const knobAngle = volumeAngleFromValue(volumeValue);
  const percentage = Math.round(volumeValue * 100);

  volumeFace.style.setProperty("--volume-angle", `${knobAngle}deg`);
  volumeKnob.setAttribute("aria-valuenow", String(percentage));
  volumeKnob.setAttribute("aria-valuetext", `${percentage} percent`);

  if (emitEvent) {
    window.dispatchEvent(new CustomEvent("listeningforward:volume", {
      detail: { value: volumeValue }
    }));
  }
}

volumeKnob.addEventListener("pointerdown", event => {
  volumeDragging = true;
  lastVolumePointerAngle = normaliseSignedAngle(
    pointerAngle(event, volumeKnob)
  );
  volumeKnob.setPointerCapture(event.pointerId);
});

volumeKnob.addEventListener("pointermove", event => {
  if (!volumeDragging || lastVolumePointerAngle === null) return;

  const currentPointerAngle = normaliseSignedAngle(
    pointerAngle(event, volumeKnob)
  );

  const delta = shortestAngularDelta(
    currentPointerAngle,
    lastVolumePointerAngle
  );

  const currentKnobAngle = volumeAngleFromValue(volumeValue);
  const nextKnobAngle = clamp(
    currentKnobAngle + delta,
    VOLUME_MIN_ANGLE,
    VOLUME_MAX_ANGLE
  );

  lastVolumePointerAngle = currentPointerAngle;
  updateVolume(volumeValueFromAngle(nextKnobAngle));
});

volumeKnob.addEventListener("pointerup", event => {
  volumeDragging = false;
  lastVolumePointerAngle = null;

  if (volumeKnob.hasPointerCapture(event.pointerId)) {
    volumeKnob.releasePointerCapture(event.pointerId);
  }
});

volumeKnob.addEventListener("pointercancel", () => {
  volumeDragging = false;
  lastVolumePointerAngle = null;
});

volumeKnob.addEventListener("wheel", event => {
  event.preventDefault();

  const direction = event.deltaY < 0 ? 1 : -1;
  updateVolume(volumeValue + direction * 0.03);
}, { passive: false });

volumeKnob.addEventListener("keydown", event => {
  const step = event.shiftKey ? 0.01 : 0.04;

  if (event.key === "ArrowRight" || event.key === "ArrowUp") {
    event.preventDefault();
    updateVolume(volumeValue + step);
  }

  if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
    event.preventDefault();
    updateVolume(volumeValue - step);
  }

  if (event.key === "Home") {
    event.preventDefault();
    updateVolume(0);
  }

  if (event.key === "End") {
    event.preventDefault();
    updateVolume(1);
  }
});

updateDial(angle);
updateVolume(volumeValue, false);
