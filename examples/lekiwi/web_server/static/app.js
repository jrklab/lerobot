"use strict";

// --- WebSocket connection, with auto-reconnect ---

let ws = null;
let currentSpeed = "medium";
let lastKnownMode = null;
// Matches the server's 30Hz control loop (robot_bridge.py FPS), well under the 400ms watchdog.
const SEND_INTERVAL_MS = 1000 / 30;
const DESKTOP = window.matchMedia("(min-width: 1024px) and (min-height: 600px)");

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
        updateJointLoads(msg.joints);
        updateStallTone(msg.joints);
        updateModeUI(msg.control_mode, msg.auto);
        updateRecordingUI(msg.recording, msg.internet_reachable);
        updateStatus(msg);
      } else if (msg.type === "error") {
        showToast(msg.message, true);
      }
    } catch (_) {
      /* ignore malformed status messages */
    }
  });
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// One-off notifications: server-reported errors (otherwise invisible: the UI would just not
// change) and the STOP's confirmation.
let toastTimer = null;

function showToast(message, isError = false) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.toggle("error", isError);
  el.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("visible"), 4000);
}

// --- Status: the top bar's badges and the Status pane ---

function setBadge(id, text, level) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = "badge" + (level ? " " + level : "");
}

function setStatus(id, text, level) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = level || "";
}

function setConnStatus(connected) {
  setBadge("conn-status", connected ? "online" : "offline", connected ? "good" : "bad");
  setStatus("st-link", connected ? "connected" : "disconnected: reconnecting…", connected ? "good" : "bad");
}

function updateStatus(msg) {
  const cal = msg.arm_calibration;
  const calLevel = cal === "calibrated" ? "good" : cal ? "warn" : "";
  setBadge("arm-badge", cal ? `arm ${cal}` : "arm --", calLevel);
  setStatus("st-arm", cal || "no arm", calLevel);

  // The servo link: failing while a motor transfer failed in the last 2 s (robot_bridge.health).
  const h = msg.health || {};
  if (h.bus_ok === false) {
    setBadge("bus-badge", "servos FAILING", "bad");
    setStatus("st-bus", `failing (last error ${h.bus_error_age_s} s ago)`, "bad");
  } else {
    setBadge("bus-badge", "servos ok", "good");
    setStatus("st-bus", h.bus_error_age_s == null ? "ok" : `ok (last error ${fmtAge(h.bus_error_age_s)} ago)`, "good");
  }
  document.getElementById("status-dot").hidden = h.bus_ok !== false;

  const cmd = msg.auto && msg.auto.command;
  if (!cmd) {
    setStatus("st-auto", "no command yet", "");
  } else {
    const level = cmd.state === "running" ? "good" : cmd.state === "done" ? "" : "warn";
    setStatus("st-auto", `${cmd.kind}: ${cmd.state}${cmd.reason ? " (" + cmd.reason + ")" : ""}`, level);
  }

  const grip = (msg.joints || {}).arm_gripper || {};
  const hold = msg.auto && msg.auto.hold;
  const parts = [hold ? `hold: ${hold}` : "not holding"];
  if (grip.temp != null) parts.push(`${Math.round(grip.temp)} °C`);
  setStatus("st-grip", parts.join(", "), "");

  setStatus("st-mode", modeName(msg.control_mode), "");
  setStatus("st-gamepad", msg.gamepad_connected ? "connected" : "not connected", msg.gamepad_connected ? "good" : "");
  setStatus("st-internet", msg.internet_reachable ? "reachable" : "not reachable", msg.internet_reachable ? "good" : "");
}

function fmtAge(s) {
  return s < 120 ? `${Math.round(s)} s` : s < 7200 ? `${Math.round(s / 60)} min` : `${Math.round(s / 3600)} h`;
}

const MODE_NAMES = { gamepad: "Gamepad", web: "Webpage", leader_keyboard: "Leader arm", vr: "VR (Quest)", auto: "Auto (brain)" };

function modeName(mode) {
  return MODE_NAMES[mode] || mode || "--";
}

// Per-joint load (Present_Load is 0.1% of full scale: 1000 = full torque) and stall coloring
// (robot_bridge.py's _stall_severity, the condition in examples/lekiwi/torque_feedback.md).
function updateJointLoads(joints) {
  if (!joints) return;
  document.querySelectorAll(".jog-card").forEach((card) => {
    const info = joints[card.dataset.joint];
    if (!info) return;
    const pct = Math.min(100, Math.abs(info.load) / 10);
    card.querySelector(".jog-load").textContent = `${Math.round(pct)}%`;
    card.querySelector(".load-bar i").style.width = `${pct}%`;
    card.classList.toggle("stalled", !!info.stalled);
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

// --- Control mode and speed (segmented buttons) ---

function setSegActive(segId, attr, value) {
  document.querySelectorAll(`#${segId} button`).forEach((b) => b.classList.toggle("active", b.dataset[attr] === value));
}

function setupModeAndSpeed() {
  document.querySelectorAll("#mode-seg button").forEach((b) => {
    b.addEventListener("click", () => send({ type: "set_mode", mode: b.dataset.mode }));
  });
  document.querySelectorAll("#speed-seg button").forEach((b) => {
    b.addEventListener("click", () => {
      currentSpeed = b.dataset.speed;
      setSegActive("speed-seg", "speed", currentSpeed);
    });
  });
  setSegActive("speed-seg", "speed", currentSpeed);
}

// The page's controls only act in "web" mode (the server ignores the rest): greyed out
// otherwise. VR and Auto are entered elsewhere and shown as a pill.
function updateModeUI(mode, auto) {
  if (mode && mode !== lastKnownMode) {
    lastKnownMode = mode;
    document.body.classList.toggle("locked", mode !== "web");
    setSegActive("mode-seg", "mode", mode);
    if (mode !== "web") releaseAll();
  }
  const extra = document.getElementById("mode-extra");
  if (mode === "vr") {
    extra.hidden = false;
    extra.textContent = "VR is driving";
  } else if (mode === "auto") {
    const cmd = auto && auto.command;
    extra.hidden = false;
    extra.textContent = cmd && cmd.state === "running" ? `Auto: ${cmd.kind}` : "Auto: idle";
  } else {
    extra.hidden = true;
  }
}

// --- STOP: the base and arm jogs halted, Auto ended, control back to the gamepad ---

function emergencyStop() {
  releaseAll();
  send({ type: "emergency_stop", to_gamepad: true });
  showToast("Stopped. Control is back on the gamepad.");
}

// --- Tabs (phone) and cameras ---

let camChoice = { drive: "front", arm: "wrist", rec: "front" }; // the phone's camera per tab

function currentTab() {
  return document.body.dataset.tab;
}

function showTab(tab) {
  document.querySelectorAll(".tabbar button").forEach((x) => x.classList.toggle("active", x.dataset.tab === tab));
  document.body.dataset.tab = tab;
  history.replaceState(null, "", `#${tab}`); // a tab can be bookmarked: /#arm
  releaseAll();
  updateVideoStreams();
}

function setupTabs() {
  document.querySelectorAll(".tabbar button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const start = location.hash.slice(1);
  if (document.querySelector(`.tabbar button[data-tab="${start}"]`)) showTab(start);
  document.querySelectorAll(".cam-switch button").forEach((b) => {
    b.addEventListener("click", () => {
      camChoice[currentTab()] = b.dataset.cam;
      updateVideoStreams();
    });
  });
  DESKTOP.addEventListener("change", updateVideoStreams);
}

// Only the feeds on screen stream (MJPEG costs the Pi encode time and bandwidth): on a desktop
// the front and the wrist camera; on a phone the one picked for the tab, none on Status.
function setSrc(img, src) {
  if (img.getAttribute("src") !== src) img.setAttribute("src", src);
}

function updateVideoStreams() {
  const main = document.getElementById("cam-main");
  const side = document.getElementById("cam-side");
  if (DESKTOP.matches) {
    setSrc(main, "/video/front");
    setSrc(side, "/video/wrist");
    return;
  }
  setSrc(side, "");
  const tab = currentTab();
  if (tab === "status") {
    setSrc(main, "");
    return;
  }
  const cam = camChoice[tab] || "front";
  setSrc(main, `/video/${cam}`);
  document.querySelectorAll(".cam-switch button").forEach((b) => b.classList.toggle("active", b.dataset.cam === cam));
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
    el.addEventListener("pointerup", () => this.release());
    el.addEventListener("pointercancel", () => this.release());
    el.addEventListener("pointerleave", () => this.release());
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

  release() {
    if (!this.active) return;
    this.active = false;
    this.x = 0;
    this.y = 0;
    this.knob.style.transform = "translate(-50%, -50%)";
    this.onChange(0, 0);
  }
}

// --- Driving: joysticks, or the keyboard on a desktop (W/S, A/D, Q/E while held) ---

let translate = { x: 0, y: 0 };
let rotate = { x: 0, y: 0 };
const joysticks = [];
const keysDown = new Set();
const DRIVE_KEYS = new Set(["w", "a", "s", "d", "q", "e"]);

function typing(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}

function setupDriving() {
  joysticks.push(new Joystick(document.getElementById("joy-translate"), (x, y) => (translate = { x, y })));
  joysticks.push(new Joystick(document.getElementById("joy-rotate"), (x) => (rotate = { x, y: 0 })));

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" || (e.key === " " && !typing(e))) {
      e.preventDefault();
      emergencyStop();
      return;
    }
    const k = e.key.toLowerCase();
    if (!DRIVE_KEYS.has(k) || typing(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    if (lastKnownMode === "web") keysDown.add(k);
  });
  window.addEventListener("keyup", (e) => keysDown.delete(e.key.toLowerCase()));
  window.addEventListener("blur", () => keysDown.clear());
  document.addEventListener("visibilitychange", () => document.hidden && releaseAll());

  // Always send, even when idle/zero: the server's watchdog relies on a steady stream
  // of "this is the true current state" messages. Skipping sends while at rest meant
  // releasing the joystick never actually told the server to stop (see robot_bridge.py's
  // per-input-source watchdog timestamps).
  setInterval(() => {
    let x = translate.y; // forward/back
    let y = -translate.x; // left/right (screen-right should strafe right)
    let theta = -rotate.x; // twist joystick left/right => rotate
    if (keysDown.size) {
      const k = (c) => (keysDown.has(c) ? 1 : 0);
      x = k("w") - k("s");
      y = k("a") - k("d");
      theta = k("q") - k("e");
    }
    send({ type: "base", x, y, theta, speed: currentSpeed });
  }, SEND_INTERVAL_MS);
}

// --- Arm jog buttons (hold-to-jog, Pointer Events) ---

const activeJogs = new Map(); // joint -> dir

function setupJogButtons() {
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

// Everything the page is holding down, let go (tab switch, mode change, STOP, hidden page).
function releaseAll() {
  joysticks.forEach((j) => j.release());
  keysDown.clear();
  activeJogs.forEach((_, joint) => send({ type: "jog", joint, dir: 0, speed: currentSpeed }));
  activeJogs.clear();
}

// "Reset arm to rest" needs a 0.8 s hold: a stray tap mustn't swing the arm home.
const HOLD_MS = 800;

function setupResetButton() {
  const btn = document.getElementById("arm-reset-btn");
  let timer = null;
  const cancel = () => {
    clearTimeout(timer);
    timer = null;
    btn.classList.remove("holding");
  };
  btn.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    btn.setPointerCapture(e.pointerId);
    btn.classList.add("holding");
    timer = setTimeout(() => {
      send({ type: "reset_arm" });
      showToast("Arm going to rest.");
      cancel();
    }, HOLD_MS);
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) => btn.addEventListener(ev, cancel));
}

// --- Record tab: episode recording, playback, and Hub upload ---

let lastEpisodeCount = -1; // only re-render the episode list when it actually changes

function setupRecordingUI() {
  const toggleBtn = document.getElementById("rec-toggle-btn");
  const discardBtn = document.getElementById("rec-discard-btn");
  const uploadBtn = document.getElementById("upload-btn");
  const taskInput = document.getElementById("rec-task");
  const episodeList = document.getElementById("episode-list");

  toggleBtn.addEventListener("click", () => {
    if (toggleBtn.classList.contains("is-recording")) {
      send({ type: "stop_recording" });
    } else {
      const task = taskInput.value.trim();
      if (!task) {
        taskInput.focus();
        return;
      }
      send({ type: "start_recording", task });
    }
  });

  discardBtn.addEventListener("click", () => send({ type: "discard_recording" }));
  uploadBtn.addEventListener("click", () => send({ type: "upload_to_hub" }));

  // Event delegation: episode rows are re-rendered on every list change, so a single
  // listener on the container avoids having to re-attach/leak per-row handlers.
  episodeList.addEventListener("click", (e) => {
    const playBtn = e.target.closest(".episode-play-btn");
    if (playBtn) {
      const episode = parseInt(playBtn.dataset.episode, 10);
      if (playBtn.classList.contains("is-playing")) {
        send({ type: "stop_playback" });
      } else {
        send({ type: "start_playback", episode });
      }
      return;
    }
    const deleteBtn = e.target.closest(".episode-delete-btn");
    if (deleteBtn) {
      const episode = parseInt(deleteBtn.dataset.episode, 10);
      if (confirm(`Delete episode ${episode}? This cannot be undone.`)) {
        send({ type: "delete_episode", episode });
      }
    }
  });
}

function updateRecordingUI(recording, internetReachable) {
  setBadge("internet-status", internetReachable ? "internet ok" : "no internet", internetReachable ? "good" : "");
  if (!recording) return;

  const toggleBtn = document.getElementById("rec-toggle-btn");
  const discardBtn = document.getElementById("rec-discard-btn");
  const taskInput = document.getElementById("rec-task");
  const liveStatus = document.getElementById("rec-live-status");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadStatus = document.getElementById("upload-status");
  const episodeList = document.getElementById("episode-list");

  const busy = recording.recording || recording.saving || recording.playback_active;

  if (recording.recording) {
    toggleBtn.textContent = "■ Stop recording";
    toggleBtn.classList.add("is-recording");
    taskInput.value = recording.task;
    liveStatus.hidden = false;
    liveStatus.textContent = `● REC  ${recording.elapsed_s.toFixed(1)}s  (${recording.frame_count} frames)`;
  } else if (recording.saving) {
    toggleBtn.textContent = "Saving…";
    toggleBtn.classList.remove("is-recording");
    liveStatus.hidden = false;
    liveStatus.textContent = "Saving episode…";
  } else {
    toggleBtn.textContent = "● Record";
    toggleBtn.classList.remove("is-recording");
    liveStatus.hidden = true;
  }
  toggleBtn.disabled = recording.saving || recording.playback_active;
  taskInput.disabled = recording.recording || recording.saving;
  discardBtn.hidden = !recording.recording;

  const canUpload = !busy && recording.episodes.length > 0 && recording.upload_status !== "uploading";
  uploadBtn.disabled = !canUpload;
  uploadBtn.textContent = recording.upload_status === "uploading" ? "Uploading…" : "↑ Upload all to Hub";
  uploadStatus.textContent = recording.upload_message || "";
  uploadStatus.className =
    "rec-upload-status" +
    (recording.upload_status === "success" ? " upload-success" : "") +
    (recording.upload_status === "error" ? " upload-error" : "");

  // Full re-render only when the episode set itself changes (a new one saved); every other
  // tick just syncs each row's play/stop button in place, to avoid flicker on every 0.2s
  // status update while a replay is running.
  if (recording.episodes.length !== lastEpisodeCount) {
    lastEpisodeCount = recording.episodes.length;
    renderEpisodeList(recording, episodeList);
  }
  episodeList.querySelectorAll(".episode-play-btn").forEach((btn) => {
    const isThis =
      recording.playback_active && parseInt(btn.dataset.episode, 10) === recording.playback_episode;
    btn.classList.toggle("is-playing", isThis);
    btn.textContent = isThis ? "■ Stop" : "▶ Play";
    btn.disabled = recording.recording || recording.saving || (recording.playback_active && !isThis);
  });
  episodeList.querySelectorAll(".episode-delete-btn").forEach((btn) => {
    const isThis =
      recording.playback_active && parseInt(btn.dataset.episode, 10) === recording.playback_episode;
    btn.disabled = recording.recording || recording.saving || isThis;
  });
}

function renderEpisodeList(recording, container) {
  if (recording.episodes.length === 0) {
    container.innerHTML = '<p class="stub-note">No episodes recorded yet.</p>';
    return;
  }
  container.innerHTML = recording.episodes
    .map((ep) => {
      const isPlaying = recording.playback_active && recording.playback_episode === ep.index;
      const playDisabled = recording.recording || recording.saving || (recording.playback_active && !isPlaying);
      const deleteDisabled = recording.recording || recording.saving || isPlaying;
      return `
        <div class="episode-row">
          <div class="episode-row-info">
            <span class="episode-row-task">Episode ${ep.index}: ${escapeHtml(ep.task)}</span>
            <span class="episode-row-meta">${ep.duration_s.toFixed(1)}s &middot; ${ep.length} frames</span>
          </div>
          <button class="episode-play-btn${isPlaying ? " is-playing" : ""}" data-episode="${ep.index}" ${playDisabled ? "disabled" : ""}>
            ${isPlaying ? "■ Stop" : "▶ Play"}
          </button>
          <button class="episode-delete-btn" data-episode="${ep.index}" ${deleteDisabled ? "disabled" : ""} title="Delete episode" aria-label="Delete episode">
            &#128465;
          </button>
        </div>`;
    })
    .join("");
}

// --- The cameras' image controls (Status tab; server camera_controls.py) ---

const CAM_CTL_LABELS = {
  auto_exposure: "Exposure mode",
  exposure_time_absolute: "Exposure time (x0.1 ms)",
  exposure_dynamic_framerate: "Slower in dim light", // auto exposure may drop the frame rate
  brightness: "Brightness",
  gain: "Gain",
  backlight_compensation: "Backlight compensation",
  white_balance_automatic: "Auto white balance",
  power_line_frequency: "Mains flicker",
};
const CAM_CTL_SEND_MS = 150; // a slider's value goes out at most this often while it moves

function camCtlRender(block, controls) {
  const body = block.querySelector(".cam-ctl-body");
  body.innerHTML = "";
  for (const [name, c] of Object.entries(controls)) {
    const label = document.createElement("label");
    label.textContent = CAM_CTL_LABELS[name] || name;
    let input;
    const out = document.createElement("output");
    if (c.menu) {
      input = document.createElement("select");
      for (const [v, text] of Object.entries(c.menu)) input.add(new Option(text, v, false, Number(v) === c.value));
    } else if (c.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.checked = c.value === 1;
    } else {
      input = document.createElement("input");
      input.type = "range";
      Object.assign(input, { min: c.min, max: c.max, step: c.step, value: c.value });
      out.textContent = c.value;
      input.addEventListener("input", () => (out.textContent = input.value));
    }
    // Inactive: set by the camera itself (the exposure time under auto exposure).
    input.disabled = c.inactive;
    [label, input, out].forEach((el) => el.classList.toggle("off", c.inactive));
    const value = () => (input.type === "checkbox" ? (input.checked ? 1 : 0) : Number(input.value));
    let timer = null;
    const sendNow = () => {
      timer = null;
      camCtlSend(block, { [name]: value() });
    };
    input.addEventListener(input.type === "range" ? "input" : "change", () => {
      if (!timer) timer = setTimeout(sendNow, CAM_CTL_SEND_MS);
    });
    body.append(label, input, out);
  }
}

async function camCtlRequest(block, url, body) {
  const state = block.querySelector(".cam-ctl-state");
  try {
    const r = await fetch(url, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!j.ok) throw new Error(j.reason || r.statusText);
    state.textContent = "";
    camCtlRender(block, j.controls);
  } catch (e) {
    state.textContent = "not available";
    showToast(`${block.dataset.cam} camera: ${e.message}`, true);
  }
}

function camCtlSend(block, values) {
  return camCtlRequest(block, `/camera/${block.dataset.cam}/controls`, values);
}

function setupCameraControls() {
  document.querySelectorAll(".cam-ctl").forEach((block) => {
    const cam = block.dataset.cam;
    const preview = block.querySelector(".cam-ctl-preview");
    block.addEventListener("toggle", () => {
      setSrc(preview, block.open ? `/video/${cam}` : "");
      if (block.open) camCtlRequest(block, `/camera/${cam}/controls`);
    });
    block.querySelector(".cam-ctl-reset").addEventListener("click", () =>
      camCtlRequest(block, `/camera/${cam}/controls/reset`, {}));
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("stop-btn").addEventListener("click", emergencyStop);
  setupTabs();
  setupModeAndSpeed();
  setupDriving();
  setupJogButtons();
  setupResetButton();
  setupRecordingUI();
  setupCameraControls();
  updateVideoStreams();
  connectWs();
});
