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
        // Live joint readout could be rendered here later (e.g. in a diagnostics tab).
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
