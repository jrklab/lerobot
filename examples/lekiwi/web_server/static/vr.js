"use strict";

// Quest 2 VR teleop client: displays the front/wrist MJPEG camera feeds (same streams
// static/app.js's flat UI uses) as two flat WebGL-textured panels inside a WebXR immersive
// session, and drives the robot via two controller modes -- see the "Controller input"
// section below for the full button mapping, robot_bridge.py's update_vr_pose()/
// _apply_vr_target() for the IK-mode server-side half, and CONTROLS.md's "Quest 2 VR"
// section for user-facing docs.
//
// No external libraries (no three.js, no CDN) -- raw WebGL/WebXR only, since the Pi's own
// hotspot network (which the Quest joins to reach this page) has no internet route, so
// anything pulled from a CDN would simply fail to load.
//
// Hand assignment is hardcoded (right hand always drives the arm, left always drives the
// base/e-stop) -- not configurable in v1.

// --- WebSocket (same /ws/control endpoint the main app uses) ---

let ws = null;
const SEND_INTERVAL_MS = 1000 / 30; // matches robot_bridge.py's 30Hz control loop

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/control`);
  ws.addEventListener("open", () => {
    setWsStatus("Connected. Entering VR mode...");
    send({ type: "set_mode", mode: "vr" });
  });
  ws.addEventListener("close", () => {
    setWsStatus("Disconnected -- retrying...");
    setTimeout(connectWs, 1000);
  });
  ws.addEventListener("error", () => ws.close());
  ws.addEventListener("message", (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") handleStallHaptics(msg.joints);
    } catch (e) {
      // Ignore malformed/unexpected messages -- this client only cares about "state".
    }
  });
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// Separate status lines for the WebSocket connection and WebXR support -- sharing one
// element meant the (usually-fast) WebSocket "connected" message could overwrite a WebXR
// support error that arrived first, hiding the real reason "Enter VR" was greyed out.
function setWsStatus(text) {
  const el = document.getElementById("ws-status");
  if (el) el.textContent = text;
}

function setXrStatus(text) {
  const el = document.getElementById("xr-status");
  if (el) el.textContent = text;
}

connectWs();

// --- WebXR session lifecycle ---

const canvas = document.getElementById("gl-canvas");
const enterBtn = document.getElementById("enter-vr-btn");
const camFrontImg = document.getElementById("cam-front");
const camWristImg = document.getElementById("cam-wrist");

let gl = null;
let xrSession = null;
let xrRefSpace = null;

async function checkSupport() {
  // WebXR requires a "secure context" (HTTPS, or localhost) -- a plain http:// page on a
  // LAN hostname/IP (which is all this server can offer, since the Pi's hotspot has no
  // route to get a real TLS cert) doesn't qualify. This is the most likely reason
  // navigator.xr is missing or isSessionSupported() returns false, so check and report it
  // explicitly rather than leaving it looking like a generic "not supported" device issue.
  if (!window.isSecureContext) {
    setXrStatus(
      "Not a secure context (need HTTPS or localhost) -- WebXR is unavailable over plain HTTP. See CONTROLS.md's Quest VR troubleshooting section."
    );
    enterBtn.disabled = true;
    return;
  }
  if (!navigator.xr) {
    setXrStatus("WebXR not available in this browser.");
    enterBtn.disabled = true;
    return;
  }
  const supported = await navigator.xr.isSessionSupported("immersive-vr");
  setXrStatus(supported ? "Ready." : "immersive-vr not supported on this device.");
  enterBtn.disabled = !supported;
}
checkSupport();

enterBtn.addEventListener("click", async () => {
  try {
    xrSession = await navigator.xr.requestSession("immersive-vr", {
      requiredFeatures: ["local-floor"],
    });
  } catch (e) {
    setXrStatus(`Failed to start VR session: ${e}`);
    return;
  }

  gl = canvas.getContext("webgl", { xrCompatible: true });
  await gl.makeXRCompatible();
  xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
  xrRefSpace = await xrSession.requestReferenceSpace("local-floor");

  setupGL();
  camFrontImg.src = "/video/front";
  camWristImg.src = "/video/wrist";

  xrSession.addEventListener("end", onSessionEnd);
  xrSession.requestAnimationFrame(onXRFrame);
  document.getElementById("overlay").style.display = "none";
});

function onSessionEnd() {
  xrSession = null;
  camFrontImg.src = "";
  camWristImg.src = "";
  document.getElementById("overlay").style.display = "";
}

// --- GL program / geometry / textures ---

const VERT_SRC = `
attribute vec3 aPosition;
attribute vec2 aTexCoord;
uniform mat4 uMVP;
varying vec2 vTexCoord;
void main() {
  gl_Position = uMVP * vec4(aPosition, 1.0);
  vTexCoord = aTexCoord;
}`;

const FRAG_SRC = `
precision mediump float;
varying vec2 vTexCoord;
uniform sampler2D uTexture;
void main() {
  gl_FragColor = texture2D(uTexture, vTexCoord);
}`;

let program, posLoc, texLoc, mvpLoc, samplerLoc;
let quadBuffer, texCoordBuffer;
let frontTexture, wristTexture;

function compileShader(type, src) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(info);
  }
  return shader;
}

function setupGL() {
  const vs = compileShader(gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);
  program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }
  posLoc = gl.getAttribLocation(program, "aPosition");
  texLoc = gl.getAttribLocation(program, "aTexCoord");
  mvpLoc = gl.getUniformLocation(program, "uMVP");
  samplerLoc = gl.getUniformLocation(program, "uTexture");

  // Unit-ish quad (0.8m x 0.6m, 4:3 to match the cameras' 640x480 frames) in its own local
  // space (XY plane, Z=0); each panel is placed/spaced via its model matrix in drawScene().
  quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-0.4, -0.3, 0, 0.4, -0.3, 0, -0.4, 0.3, 0, 0.4, 0.3, 0]),
    gl.STATIC_DRAW
  );
  texCoordBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  // V flipped (0=top) so the MJPEG frame isn't rendered upside down.
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 1, 1, 1, 0, 0, 1, 0]), gl.STATIC_DRAW);

  frontTexture = createTexture();
  wristTexture = createTexture();
  gl.clearColor(0.05, 0.05, 0.05, 1.0);
  gl.enable(gl.DEPTH_TEST);
}

function createTexture() {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // 1x1 placeholder until the first camera frame arrives.
  gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([40, 40, 40, 255])
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return tex;
}

function updateTextureFromImage(tex, imgEl) {
  if (!imgEl.complete || imgEl.naturalWidth === 0) return;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  try {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, imgEl);
  } catch (e) {
    // Rarely races with the MJPEG stream swapping to a new frame mid-decode; just skip
    // this frame's texture update and try again next frame.
  }
}

// --- mat4 helpers (column-major, matching WebGL/WebXR's convention -- no external library) ---

function mat4Multiply(out, a, b) {
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

function mat4Translate(x, y, z) {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}

// --- Render loop ---

function onXRFrame(time, frame) {
  const session = frame.session;
  session.requestAnimationFrame(onXRFrame);

  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose) return;

  updateTextureFromImage(frontTexture, camFrontImg);
  updateTextureFromImage(wristTexture, camWristImg);
  handleInput(frame);

  const glLayer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  for (const view of pose.views) {
    const viewport = glLayer.getViewport(view);
    gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
    drawScene(view);
  }
}

function drawScene(view) {
  gl.useProgram(program);

  const viewProj = mat4Multiply(new Float32Array(16), view.projectionMatrix, view.transform.inverse.matrix);

  // Two panels side by side, ~1.2m in front of and slightly below eye height (local-floor
  // space is +Y up from the floor) -- front camera on the left, wrist camera on the right.
  drawQuad(viewProj, mat4Translate(-0.45, 1.4, -1.2), frontTexture);
  drawQuad(viewProj, mat4Translate(0.45, 1.4, -1.2), wristTexture);
}

function drawQuad(viewProj, modelMatrix, texture) {
  const mvp = mat4Multiply(new Float32Array(16), viewProj, modelMatrix);
  gl.uniformMatrix4fv(mvpLoc, false, mvp);

  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
  gl.enableVertexAttribArray(texLoc);
  gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 0, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(samplerLoc, 0);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// --- Controller input ---
//
// Button indices follow the "xr-standard" WebXR gamepad mapping (Quest Touch controllers):
// buttons[0]=trigger, buttons[1]=squeeze/grip, buttons[3]=thumbstick click, buttons[4]=lower
// face button (A on right controller, X on left), buttons[5]=upper face button (B on right,
// Y on left); axes[2]/axes[3]=thumbstick X/Y.
//
// Two top-level modes, toggled by the right thumbstick click:
//   - "ik": right hand drives the arm by absolute pose + clutch (see update_vr_pose()'s
//     docstring); left hand drives the base (stick=translate, X/Y=rotate). This is the
//     original VR control scheme.
//   - "motor" (default on entry): direct joint/base jogging, mirroring the gamepad exactly
//     (gamepad_input.py's _dispatch_base()/_dispatch_arm()) but split across two hands
//     instead of one gamepad's two sticks. Has its own base/arm sub-mode, toggled by the
//     left thumbstick click (mirrors the gamepad's joint/cartesian sub-mode toggle).
//
// Always active regardless of mode (mirrors the gamepad's "always-active controls"): right
// A cycles speed, right B resets the arm to neutral, left grip is emergency stop. Left grip
// specifically (not a trigger) because Motor Control's arm sub-mode needs both triggers for
// the gripper, leaving no trigger free -- grip is unused by every other mode/sub-mode, so it
// works identically and always, exactly like the gamepad's RB.

const SPEED_NAMES = ["slow", "medium", "fast"];
let speedIndex = 1; // start at medium, matching the gamepad's default
let vrMode = "motor"; // "ik" | "motor"
let armSubmode = "base"; // "base" | "arm" -- only meaningful within "motor" mode

function cycleSpeed() {
  speedIndex = (speedIndex + 1) % SPEED_NAMES.length;
  const el = document.getElementById("speed-status");
  if (el) el.textContent = `Speed: ${SPEED_NAMES[speedIndex]}`;
}

function updateModeStatus() {
  const el = document.getElementById("mode-status");
  if (!el) return;
  el.textContent =
    vrMode === "ik" ? "Mode: IK (arm) + base" : `Mode: Motor Control -- ${armSubmode}`;
}
updateModeStatus();

const ARM_JOINTS = [
  "arm_shoulder_pan",
  "arm_shoulder_lift",
  "arm_elbow_flex",
  "arm_wrist_flex",
  "arm_wrist_roll",
  "arm_gripper",
];

// Zeroes any in-progress jog/base command server-side, so switching modes/sub-modes can't
// leave a stale command latched in (mirrors gamepad_input.py's mode-toggle handlers, which
// do the same before switching).
function releaseAllJogs() {
  for (const joint of ARM_JOINTS) {
    send({ type: "jog", joint: joint, dir: 0, speed: SPEED_NAMES[speedIndex], source: "vr" });
  }
  send({ type: "base", x: 0, y: 0, theta: 0, speed: SPEED_NAMES[speedIndex], source: "vr" });
}

// Tracks each button's previous-frame pressed state so edge-triggered actions (mode
// toggles, speed cycle, reset) fire once per press rather than every frame it's held.
const prevButtonState = new Map();
function pressedEdge(key, pressed) {
  const was = prevButtonState.get(key) || false;
  prevButtonState.set(key, pressed);
  return pressed && !was;
}

// --- Stall haptics ---
//
// The server already broadcasts each arm joint's stall severity (0..1, see
// robot_bridge.py's _stall_severity()) to every connected client every 200ms -- the same
// data the flat web app's per-joint load readout uses. Reused here as-is (no server changes
// needed): pulse the right controller (the one driving the arm, in both VR modes) at an
// intensity matching the worst-stalled joint, so a motor straining under load is something
// you can feel, not just something you'd have to notice on a screen.

const STALL_HAPTIC_PULSE_MS = 150; // matches (with margin) the 200ms "state" broadcast
// interval, so a sustained stall feels like continuous buzzing rather than distinct taps.

function handleStallHaptics(joints) {
  if (!joints || !xrSession) return;
  let maxSeverity = 0;
  for (const key of Object.keys(joints)) {
    const j = joints[key];
    if (j && typeof j.severity === "number" && j.severity > maxSeverity) maxSeverity = j.severity;
  }
  if (maxSeverity <= 0) return;

  for (const inputSource of xrSession.inputSources) {
    if (inputSource.handedness !== "right") continue;
    const gamepad = inputSource.gamepad;
    const actuator = gamepad && gamepad.hapticActuators && gamepad.hapticActuators[0];
    if (!actuator) continue;
    if (typeof actuator.pulse === "function") {
      actuator.pulse(maxSeverity, STALL_HAPTIC_PULSE_MS);
    } else if (typeof actuator.playEffect === "function") {
      // Fallback for implementations that only expose the newer effect-based API.
      actuator.playEffect("dual-rumble", {
        duration: STALL_HAPTIC_PULSE_MS,
        strongMagnitude: maxSeverity,
        weakMagnitude: maxSeverity,
      });
    }
  }
}

let lastSendTime = 0;

function handleInput(frame) {
  const now = performance.now();
  if (now - lastSendTime < SEND_INTERVAL_MS) return;
  lastSendTime = now;

  let leftSource = null;
  let rightSource = null;
  for (const inputSource of frame.session.inputSources) {
    if (!inputSource.gamepad) continue;
    if (inputSource.handedness === "left") leftSource = inputSource;
    else if (inputSource.handedness === "right") rightSource = inputSource;
  }
  if (!leftSource || !rightSource) return; // need both controllers tracked

  const leftPad = leftSource.gamepad;
  const rightPad = rightSource.gamepad;
  const lbtn = (i) => (leftPad.buttons[i] ? leftPad.buttons[i] : { value: 0, pressed: false });
  const rbtn = (i) => (rightPad.buttons[i] ? rightPad.buttons[i] : { value: 0, pressed: false });
  const laxes = leftPad.axes || [];
  const raxes = rightPad.axes || [];
  const lx = laxes.length > 2 ? laxes[2] : 0;
  const ly = laxes.length > 3 ? laxes[3] : 0;
  const rx = raxes.length > 2 ? raxes[2] : 0;
  const ry = raxes.length > 3 ? raxes[3] : 0;
  const speed = SPEED_NAMES[speedIndex];

  // Global emergency stop (left grip) -- matches gamepad_input.py's e-stop handling: skip
  // sending any other command this same tick, so one can't immediately re-establish motion
  // right after emergency_stop() zeroes it server-side.
  if (lbtn(1).pressed) {
    send({ type: "emergency_stop" });
    return;
  }

  if (pressedEdge("right-a", rbtn(4).pressed)) cycleSpeed();
  if (pressedEdge("right-b", rbtn(5).pressed)) send({ type: "reset_arm" });

  if (pressedEdge("right-stick-click", rbtn(3).pressed)) {
    vrMode = vrMode === "ik" ? "motor" : "ik";
    if (vrMode === "ik") {
      // Force-clear any clutch state left over from before this mode was last active, so
      // the next real grip press is always treated as a fresh rising edge (latching
      // wherever the arm actually is now) rather than reusing a stale reference pose from
      // however Motor Control moved the arm in between.
      send({ type: "vr_pose", position: [0, 0, 0], quaternion: [0, 0, 0, 1], grip: false, trigger: 0, speed: speed });
    }
    releaseAllJogs();
    updateModeStatus();
  }
  if (pressedEdge("left-stick-click", lbtn(3).pressed)) {
    armSubmode = armSubmode === "base" ? "arm" : "base";
    releaseAllJogs();
    updateModeStatus();
  }

  if (vrMode === "ik") {
    const gripPose = frame.getPose(rightSource.gripSpace, xrRefSpace);
    if (gripPose) {
      const p = gripPose.transform.position;
      const o = gripPose.transform.orientation;
      send({
        type: "vr_pose",
        position: [p.x, p.y, p.z],
        quaternion: [o.x, o.y, o.z, o.w],
        grip: rbtn(1).pressed,
        trigger: rbtn(0).value,
        speed: speed,
      });
    }
    const rotate = (lbtn(5).pressed ? 1 : 0) - (lbtn(4).pressed ? 1 : 0); // Y=+, X=-
    send({ type: "base", x: -ly, y: -lx, theta: rotate, speed: speed, source: "vr" });
  } else if (armSubmode === "base") {
    send({ type: "base", x: -ly, y: -lx, theta: -rx, speed: speed, source: "vr" });
  } else {
    const wristRollDir = (lbtn(5).pressed ? 1 : 0) - (lbtn(4).pressed ? 1 : 0); // Y=+, X=-
    const dirs = {
      arm_shoulder_pan: lx,
      arm_shoulder_lift: -ly,
      arm_elbow_flex: -ry,
      arm_wrist_flex: -rx,
      arm_wrist_roll: wristRollDir,
      arm_gripper: lbtn(0).value - rbtn(0).value, // left trigger=open, right trigger=close
    };
    for (const [joint, dir] of Object.entries(dirs)) {
      send({ type: "jog", joint: joint, dir: dir, speed: speed, source: "vr" });
    }
  }
}
