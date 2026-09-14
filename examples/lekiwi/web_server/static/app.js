"use strict";

// --- WebSocket connection, with auto-reconnect ---

let ws = null;
let currentSpeed = "medium";
// Matches the server's 30Hz control loop (robot_bridge.py FPS), well under the 400ms watchdog.
const SEND_INTERVAL_MS = 1000 / 30;

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/control`);

  ws.addEventListener("open", () => setConnStatus(true));
  ws.addEventListener("close", () => {
    setConnStatus(false);
    setTimeout(connectWs, 1000);
  });
  ws.addEventListener("error", () => ws.close());
  ws.addEventListener("message", (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") {
        setGamepadStatus(!!msg.gamepad_connected);
        updateJointLoads(msg.joints);
        updateStallTone(msg.joints);
      }
    } catch (_) {
      /* ignore malformed status messages */
    }
  });
}

function setConnStatus(connected) {
  const el = document.getElementById("conn-status");
  el.textContent = connected ? "connected" : "disconnected";
  el.className = "status " + (connected ? "status-connected" : "status-disconnected");
  if (!connected) setGamepadStatus(false);
}

function setGamepadStatus(connected) {
  const el = document.getElementById("gamepad-status");
  el.textContent = connected ? "gamepad: connected" : "gamepad: disconnected";
  el.className = "status " + (connected ? "status-connected" : "status-disconnected");
}

// Per-joint load readout + stall coloring (see robot_bridge.py's _stall_severity(), which
// mirrors the stall condition documented in examples/lekiwi/torque_feedback.md).
function updateJointLoads(joints) {
  if (!joints) return;
  document.querySelectorAll(".jog-card").forEach((card) => {
    const info = joints[card.dataset.joint];
    const el = card.querySelector(".jog-load");
    if (!info || !el) return;
    el.textContent = Math.round(info.load);
    el.classList.toggle("load-stall", !!info.stalled);
    el.classList.toggle("load-ok", !info.stalled);
  });
}

// --- Stall audio alert (gamepad has no vibration motor -- see CONTROLS.md -- so a beep
// substitutes for haptic feedback). Pitch/volume track the worst-stalled joint's severity
// (0-1, from robot_bridge.py) so the alert reflects how bad the stall is, not just that one
// exists. A continuous tone (not a one-shot beep) so it's always current with live severity.

let audioCtx = null;
let stallOscillator = null;
let stallGain = null;

const STALL_TONE_MIN_HZ = 300;
const STALL_TONE_MAX_HZ = 900;
const STALL_TONE_MIN_GAIN = 0.12;
const STALL_TONE_MAX_GAIN = 0.35;
const STALL_TONE_RAMP_S = 0.05;

function ensureAudioContext() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function startStallTone() {
  ensureAudioContext();
  stallOscillator = audioCtx.createOscillator();
  stallGain = audioCtx.createGain();
  stallOscillator.type = "square";
  stallGain.gain.value = 0;
  stallOscillator.connect(stallGain).connect(audioCtx.destination);
  stallOscillator.start();
}

function stopStallTone() {
  if (!stallOscillator) return;
  const osc = stallOscillator;
  const gain = stallGain;
  gain.gain.setTargetAtTime(0, audioCtx.currentTime, STALL_TONE_RAMP_S);
  setTimeout(() => {
    osc.stop();
    osc.disconnect();
    gain.disconnect();
  }, STALL_TONE_RAMP_S * 1000 * 4);
  stallOscillator = null;
  stallGain = null;
}

function updateStallTone(joints) {
  if (!joints) return;
  let maxSeverity = 0;
  for (const info of Object.values(joints)) {
    if (info.severity > maxSeverity) maxSeverity = info.severity;
  }

  if (maxSeverity <= 0) {
    stopStallTone();
    return;
  }
  if (!stallOscillator) startStallTone();
  const freq = STALL_TONE_MIN_HZ + maxSeverity * (STALL_TONE_MAX_HZ - STALL_TONE_MIN_HZ);
  const gain = STALL_TONE_MIN_GAIN + maxSeverity * (STALL_TONE_MAX_GAIN - STALL_TONE_MIN_GAIN);
  stallOscillator.frequency.setTargetAtTime(freq, audioCtx.currentTime, STALL_TONE_RAMP_S);
  stallGain.gain.setTargetAtTime(gain, audioCtx.currentTime, STALL_TONE_RAMP_S);
}

// Browsers block audio playback until a user gesture -- create/resume the AudioContext on
// the first tap/click anywhere so it's already unlocked by the time a stall can occur.
window.addEventListener("pointerdown", () => ensureAudioContext(), { once: true });

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// --- Tabs ---

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.getElementById(`tab-${tab}`).classList.add("active");
      updateVideoStreams(tab);
    });
  });
}

// Only stream from the camera whose tab is actually visible, to avoid paying MJPEG
// encode/bandwidth cost for a feed nobody's looking at.
function updateVideoStreams(activeTab) {
  const baseImg = document.getElementById("video-base");
  const armImg = document.getElementById("video-arm");
  baseImg.src = activeTab === "base" ? "/video/front" : "";
  armImg.src = activeTab === "arm" ? "/video/wrist" : "";
}

// --- Joystick widget (Pointer Events cover mouse + touch uniformly) ---

class Joystick {
  constructor(el, onChange) {
    this.el = el;
    this.knob = el.querySelector(".joystick-knob");
    this.onChange = onChange;
    this.active = false;
    this.x = 0;
    this.y = 0;

    el.addEventListener("pointerdown", (e) => this._start(e));
    el.addEventListener("pointermove", (e) => this._move(e));
    el.addEventListener("pointerup", (e) => this._end(e));
    el.addEventListener("pointercancel", (e) => this._end(e));
    el.addEventListener("pointerleave", (e) => this._end(e));
  }

  _start(e) {
    this.el.setPointerCapture(e.pointerId);
    this.active = true;
    this._move(e);
  }

  _move(e) {
    if (!this.active) return;
    const rect = this.el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = rect.width / 2;

    let dx = (e.clientX - cx) / radius;
    let dy = (e.clientY - cy) / radius;
    const mag = Math.hypot(dx, dy);
    if (mag > 1) {
      dx /= mag;
      dy /= mag;
    }
    this.x = dx;
    this.y = -dy; // screen-down is negative y; joystick "up" should be positive
    this.knob.style.transform = `translate(-50%, -50%) translate(${dx * radius}px, ${-this.y * radius}px)`;
    this.onChange(this.x, this.y);
  }

  _end(e) {
    if (!this.active) return;
    this.active = false;
    this.x = 0;
    this.y = 0;
    this.knob.style.transform = "translate(-50%, -50%)";
    this.onChange(0, 0);
  }
}

function setupBaseJoysticks() {
  let translate = { x: 0, y: 0 };
  let rotate = { x: 0, y: 0 };

  new Joystick(document.getElementById("joy-translate"), (x, y) => {
    translate = { x, y };
  });
  new Joystick(document.getElementById("joy-rotate"), (x) => {
    rotate = { x, y: 0 };
  });

  // Always send, even when idle/zero: the server's watchdog relies on a steady stream
  // of "this is the true current state" messages. Skipping sends while at rest meant
  // releasing the joystick never actually told the server to stop (see robot_bridge.py's
  // per-input-source watchdog timestamps).
  setInterval(() => {
    send({
      type: "base",
      x: translate.y, // forward/back
      y: -translate.x, // left/right (screen-right should strafe right)
      theta: -rotate.x, // twist joystick left/right => rotate
      speed: currentSpeed,
    });
  }, SEND_INTERVAL_MS);
}

// --- Arm jog buttons (hold-to-jog, Pointer Events) ---

function setupJogButtons() {
  const activeJogs = new Map(); // joint -> dir

  document.querySelectorAll(".jog-card").forEach((card) => {
    const joint = card.dataset.joint;
    card.querySelectorAll(".jog-btn").forEach((btn) => {
      const dir = parseInt(btn.dataset.dir, 10);

      const start = (e) => {
        e.preventDefault();
        btn.setPointerCapture(e.pointerId);
        activeJogs.set(joint, dir);
        send({ type: "jog", joint, dir, speed: currentSpeed });
      };
      const stop = () => {
        if (activeJogs.get(joint) === dir) {
          activeJogs.delete(joint);
          send({ type: "jog", joint, dir: 0, speed: currentSpeed });
        }
      };

      btn.addEventListener("pointerdown", start);
      btn.addEventListener("pointerup", stop);
      btn.addEventListener("pointercancel", stop);
      btn.addEventListener("pointerleave", stop);
    });
  });

  setInterval(() => {
    activeJogs.forEach((dir, joint) => {
      send({ type: "jog", joint, dir, speed: currentSpeed });
    });
  }, SEND_INTERVAL_MS);
}

function setupSpeedSelector() {
  document.querySelectorAll('input[name="speed"]').forEach((input) => {
    input.addEventListener("change", (e) => {
      if (e.target.checked) currentSpeed = e.target.value;
    });
  });
}

function setupResetButton() {
  document.getElementById("arm-reset-btn").addEventListener("click", () => {
    send({ type: "reset_arm" });
  });
}

window.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupBaseJoysticks();
  setupJogButtons();
  setupSpeedSelector();
  setupResetButton();
  updateVideoStreams("base");
  connectWs();
});
