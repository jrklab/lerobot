#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Episode recording + playback for the LeKiwi web app.

Writes a from-scratch replica of the LeRobotDataset v3.0 on-disk format (info.json,
tasks.parquet, meta/episodes/*.parquet, data/*.parquet, videos/<key>/*.mp4) WITHOUT ever
importing `lerobot.datasets` -- that package pulls in torch transitively (training-pipeline
submodules we never use), and the installed torch wheel SIGILLs the instant its native
extension loads on this Raspberry Pi's CPU (a confirmed hardware/CPU incompatibility, not
fixable from here -- see git history for the investigation). A hard, uncatchable
process-killing signal like that means the only fix is to never import anything
torch-dependent in the first place.

Confirmed by directly reading the lerobot source (not by executing it, to stay torch-free
during development too):
  - `meta/stats.json` and the per-episode `stats/*` columns are NOT required to load or
    read a dataset -- purely for training-time normalization. Safe to omit.
  - One data/video file per episode (rather than the official writer's multi-episode
    consolidation) is a fully valid layout -- each episode's own metadata row records which
    chunk/file to open, and the reader has no assumption that files hold more than one
    episode. Consolidation is purely a writer-side space optimization we don't need.
  - `tasks.parquet` is exactly `pd.DataFrame({"task_index": [...]}, index=pd.Index([...],
    name="task"))`.
  - Playback only ever reads the "action" column back -- video is write-only from this
    module's own perspective (for later human/ML review), so no video-decoding code is
    needed here at all.

Uses pandas/pyarrow (parquet), PyAV (video -- bundles its own libav, no dependency on a
system `ffmpeg` binary), numpy (stats), and `huggingface_hub`'s `HfApi` directly (upload) --
all confirmed torch-free on this hardware.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CODEBASE_VERSION = "v3.0"
DEFAULT_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"

# h264 (not the official writer's AV1 default): far cheaper to software-encode in real time
# on a Raspberry Pi 4. GOP/crf don't need to be tuned for fast seeking (unlike the official
# writer's g=2) since we never decode our own video -- any reasonable settings are fine.


class _VideoWriter:
    """Thin wrapper around a PyAV encode+mux pipeline for one camera's mp4 output."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "30"}

    def write_frame(self, rgb_array: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb_array), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():  # flush any buffered frames
            self.container.mux(packet)
        self.container.close()

    def abort(self) -> None:
        try:
            self.container.close()
        except Exception:
            logger.exception("Failed to close an aborted video writer")

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


class EpisodeRecorder:
    """Owns one dataset's recording/playback/upload lifecycle. Not thread-safe against
    concurrent calls to its mutating methods -- RobotBridge only ever calls these from
    either its own control-loop thread (add_frame) or serialized WS-message handling on
    the FastAPI event loop (everything else), never both at once for the same call."""

    def __init__(self, repo_id: str, root: str | Path | None, fps: int):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else DEFAULT_ROOT / repo_id
        self.fps = fps

        self._state_names: list[str] | None = None
        self._camera_shapes: dict[str, tuple] | None = None

        # Guards _recording/_buf_action/_buf_state/_frame_count, which add_frame() mutates
        # from the control-loop thread while stop_recording()/discard_recording() (FastAPI
        # event loop thread) read and reset them -- without this, a stop/discard racing a
        # concurrent add_frame() could capture the action/state lists mid-append, producing
        # mismatched lengths (this actually happened in production: "ValueError: All arrays
        # must be of the same length" in _save_worker, losing the episode).
        self._buffer_lock = threading.Lock()
        self._recording = False
        self._saving = False
        self._task = ""
        self._record_start_t = 0.0
        self._frame_count = 0
        self._buf_action: list[np.ndarray] = []
        self._buf_state: list[np.ndarray] = []
        # Video encoding (libx264) happens on one background thread per camera, fed through a
        # bounded queue, instead of inline in add_frame() -- encoding synchronously in the
        # control-loop thread competed with send_action() timing and made teleoperated arm
        # motion visibly jerky while recording.
        self._video_writers: dict[str, _VideoWriter] = {}
        self._video_paths: dict[str, Path] = {}
        self._frame_queues: dict[str, queue.Queue] = {}
        self._encoder_threads: dict[str, threading.Thread] = {}

        self._episodes: list[dict] = []
        self._total_frames = 0
        self._task_index: dict[str, int] = {}
        self._load_existing_metadata()

        self._upload_lock = threading.Lock()
        self._upload_status = "idle"  # "idle" | "uploading" | "success" | "error"
        self._upload_message = ""

    def configure_features(self, action_features: dict, observation_features: dict) -> None:
        """Must be called once (with the real robot's feature dicts) before recording can
        start. Deferred out of __init__ since RobotBridge only knows these after connect()."""
        self._state_names = list(action_features.keys())
        self._camera_shapes = {k: v for k, v in observation_features.items() if isinstance(v, tuple)}

    # --- metadata (re)loading ---

    def _load_existing_metadata(self) -> None:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return  # Brand new dataset -- nothing to seed.
        try:
            info = json.loads(info_path.read_text())
            self._total_frames = info.get("total_frames", 0)
        except Exception:
            logger.exception("Failed to read existing info.json at %s", info_path)

        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            try:
                df = pd.read_parquet(tasks_path)
                self._task_index = {str(task): int(idx) for task, idx in df["task_index"].items()}
            except Exception:
                logger.exception("Failed to read existing tasks.parquet at %s", tasks_path)

        episodes_dir = self.root / "meta" / "episodes"
        if episodes_dir.exists():
            try:
                rows = []
                for pq_file in sorted(episodes_dir.glob("*/*.parquet")):
                    df = pd.read_parquet(pq_file, columns=["episode_index", "tasks", "length"])
                    rows.extend(df.to_dict("records"))
                rows.sort(key=lambda r: r["episode_index"])
                self._episodes = [
                    {
                        "index": int(r["episode_index"]),
                        "task": str(r["tasks"][0]) if len(r["tasks"]) else "",
                        "length": int(r["length"]),
                        "duration_s": int(r["length"]) / self.fps,
                    }
                    for r in rows
                ]
                logger.info("Loaded %d existing episode(s) from %s", len(self._episodes), self.root)
            except Exception:
                logger.exception("Failed to load existing episode list from %s", self.root)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_saving(self) -> bool:
        return self._saving

    # --- recording ---

    def start_recording(self, task: str) -> bool:
        if self._recording or self._saving:
            return False
        if self._state_names is None or self._camera_shapes is None:
            raise RuntimeError("configure_features() must be called before recording")

        episode_index = len(self._episodes)
        self._video_writers = {}
        self._video_paths = {}
        self._frame_queues = {}
        self._encoder_threads = {}
        try:
            for cam_key, (height, width, _channels) in self._camera_shapes.items():
                out_path = (
                    self.root / "videos" / f"observation.images.{cam_key}" / "chunk-000"
                    / f"file-{episode_index:03d}.mp4"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = _VideoWriter(out_path, width, height, self.fps)
                self._video_writers[cam_key] = writer
                self._video_paths[cam_key] = out_path
                # Bounded to ~2s of frames: if encoding falls behind that much, dropping
                # frames (see add_frame()) is better than unbounded memory growth.
                q: queue.Queue = queue.Queue(maxsize=2 * self.fps)
                self._frame_queues[cam_key] = q
                thread = threading.Thread(target=self._encoder_loop, args=(cam_key, q, writer), daemon=True)
                self._encoder_threads[cam_key] = thread
                thread.start()
        except Exception:
            logger.exception("Failed to start video encoders for recording")
            self._abort_video_writers()
            return False

        with self._buffer_lock:
            self._task = task
            self._buf_action = []
            self._buf_state = []
            self._frame_count = 0
            self._recording = True
        self._record_start_t = time.monotonic()
        logger.info("Recording started: episode=%d task=%r", episode_index, task)
        return True

    def _encoder_loop(self, cam_key: str, q: queue.Queue, writer: _VideoWriter) -> None:
        """Runs on its own thread per camera so libx264 encoding never competes with the
        control loop's send_action()/get_observation() timing."""
        while True:
            frame = q.get()
            if frame is None:  # sentinel: shut down
                return
            try:
                writer.write_frame(frame)
            except Exception:
                logger.exception("Video encode failed for %s", cam_key)

    def add_frame(self, obs: dict, action: dict) -> None:
        """Called every control-loop tick while recording is active. Any failure stops the
        recording rather than crashing the caller's control loop. Fast: state/action are
        cheap numpy ops, and video frames are only handed to a queue (real encoding happens
        on the per-camera encoder threads), so this never meaningfully delays the next
        send_action() call."""
        if not self._recording:
            return  # fast path: no lock needed just to bail out when not recording
        try:
            state_arr = np.array([obs[name] for name in self._state_names], dtype=np.float32)
            action_arr = np.array([action[name] for name in self._state_names], dtype=np.float32)
            with self._buffer_lock:
                # Re-check inside the lock: stop_recording()/discard_recording() may have
                # flipped this and swapped the buffer lists out from under us since the
                # check above -- without this, we could append to lists no longer being
                # tracked by the recording that just stopped.
                if not self._recording:
                    return
                self._buf_state.append(state_arr)
                self._buf_action.append(action_arr)
                self._frame_count += 1
            for cam_key, q in self._frame_queues.items():
                frame = obs.get(cam_key)
                if frame is None:
                    continue
                try:
                    q.put_nowait(np.ascontiguousarray(frame, dtype=np.uint8))
                except queue.Full:
                    logger.warning("Video encoder queue full for %s -- dropping a frame", cam_key)
        except Exception:
            logger.exception("add_frame failed -- stopping recording")
            with self._buffer_lock:
                self._recording = False
            self._abort_video_writers()

    def _abort_video_writers(self) -> None:
        for cam_key, q in self._frame_queues.items():
            with contextlib.suppress(Exception):
                q.put_nowait(None)
        for thread in self._encoder_threads.values():
            thread.join(timeout=5)
        for writer in self._video_writers.values():
            writer.abort()
        self._video_writers = {}
        self._frame_queues = {}
        self._encoder_threads = {}

    def stop_recording(self) -> None:
        with self._buffer_lock:
            if not self._recording:
                return
            self._recording = False
            # Swap in fresh empty lists (rather than just capturing references to the old
            # ones) so it's structurally impossible for a straggling add_frame() call to
            # mutate what _save_worker is about to process -- see add_frame()'s own re-check
            # inside the lock, which is what makes that straggler a no-op in the first place.
            actions = self._buf_action
            states = self._buf_state
            self._buf_action = []
            self._buf_state = []
        episode_index = len(self._episodes)
        task = self._task
        frame_queues = self._frame_queues
        encoder_threads = self._encoder_threads
        video_writers = self._video_writers
        video_paths = self._video_paths
        # Stop accepting new frames immediately (synchronous) so add_frame() calls from the
        # control-loop thread cease at once; the slow part (draining encoder queues, closing
        # video writers, writing parquet/metadata) happens on a background thread so it never
        # blocks the FastAPI event loop for every connected client.
        self._video_writers = {}
        self._video_paths = {}
        self._frame_queues = {}
        self._encoder_threads = {}
        self._saving = True
        threading.Thread(
            target=self._save_worker,
            args=(episode_index, task, actions, states, frame_queues, encoder_threads, video_writers, video_paths),
            daemon=True,
        ).start()

    def _shutdown_encoders(
        self,
        frame_queues: dict[str, queue.Queue],
        encoder_threads: dict[str, threading.Thread],
        video_writers: dict[str, _VideoWriter],
    ) -> None:
        # Sentinel + join first: any frames already queued at stop/discard time must finish
        # encoding before we close the writer, or they'd silently be lost.
        for q in frame_queues.values():
            with contextlib.suppress(Exception):
                q.put(None)
        for cam_key, thread in encoder_threads.items():
            thread.join(timeout=30)
            if thread.is_alive():
                logger.warning("Encoder thread for %s did not finish in time", cam_key)
        for cam_key, writer in video_writers.items():
            try:
                writer.close()
            except Exception:
                logger.exception("Failed to close video writer for %s", cam_key)

    def _save_worker(
        self,
        episode_index: int,
        task: str,
        actions: list[np.ndarray],
        states: list[np.ndarray],
        frame_queues: dict[str, queue.Queue],
        encoder_threads: dict[str, threading.Thread],
        video_writers: dict[str, _VideoWriter],
        video_paths: dict[str, Path],
    ) -> None:
        frame_count = len(actions)
        try:
            self._shutdown_encoders(frame_queues, encoder_threads, video_writers)

            if frame_count == 0:
                logger.warning("Recording stopped with 0 frames -- discarding episode %d", episode_index)
                for path in video_paths.values():
                    path.unlink(missing_ok=True)
                return

            if task not in self._task_index:
                self._task_index[task] = len(self._task_index)
            task_idx = self._task_index[task]

            global_start = self._total_frames
            data = {
                "action": actions,
                "observation.state": states,
                "timestamp": [i / self.fps for i in range(frame_count)],
                "frame_index": list(range(frame_count)),
                "episode_index": [episode_index] * frame_count,
                "index": list(range(global_start, global_start + frame_count)),
                "task_index": [task_idx] * frame_count,
            }
            data_path = self.root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(data).to_parquet(data_path, index=False)

            ep_row = {
                "episode_index": episode_index,
                "tasks": [task],
                "length": frame_count,
                "data/chunk_index": 0,
                "data/file_index": episode_index,
                "dataset_from_index": global_start,
                "dataset_to_index": global_start + frame_count,
            }
            for cam_key in video_paths:
                ep_row[f"videos/observation.images.{cam_key}/chunk_index"] = 0
                ep_row[f"videos/observation.images.{cam_key}/file_index"] = episode_index
                ep_row[f"videos/observation.images.{cam_key}/from_timestamp"] = 0.0
                ep_row[f"videos/observation.images.{cam_key}/to_timestamp"] = frame_count / self.fps
            ep_meta_path = self.root / "meta" / "episodes" / "chunk-000" / f"file-{episode_index:03d}.parquet"
            ep_meta_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([ep_row]).to_parquet(ep_meta_path, index=False)

            self._write_tasks_parquet()
            self._total_frames += frame_count
            self._episodes.append(
                {"index": episode_index, "task": task, "length": frame_count, "duration_s": frame_count / self.fps}
            )
            self._write_info_json()
            logger.info("Recording saved: episode %d, %d frames", episode_index, frame_count)
        except Exception:
            logger.exception("Failed to save episode")
        finally:
            self._saving = False
            self._frame_count = 0

    def _write_tasks_parquet(self) -> None:
        tasks_path = self.root / "meta" / "tasks.parquet"
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_sorted = sorted(self._task_index.items(), key=lambda kv: kv[1])
        df = pd.DataFrame(
            {"task_index": [idx for _, idx in tasks_sorted]},
            index=pd.Index([t for t, _ in tasks_sorted], name="task"),
        )
        df.to_parquet(tasks_path)

    def _write_info_json(self) -> None:
        features = dict(DEFAULT_FEATURES)
        features["action"] = {"dtype": "float32", "shape": [len(self._state_names)], "names": self._state_names}
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [len(self._state_names)],
            "names": self._state_names,
        }
        for cam_key, (height, width, channels) in self._camera_shapes.items():
            features[f"observation.images.{cam_key}"] = {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.fps": self.fps,
                    "video.channels": channels,
                    "has_audio": False,
                },
            }
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": "lekiwi",
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._task_index),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        info_path = self.root / "meta" / "info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))

    def discard_recording(self) -> None:
        with self._buffer_lock:
            if not self._recording:
                return
            self._recording = False
            self._buf_action = []
            self._buf_state = []
        frame_queues = self._frame_queues
        encoder_threads = self._encoder_threads
        video_writers = self._video_writers
        video_paths = self._video_paths
        self._video_writers = {}
        self._video_paths = {}
        self._frame_queues = {}
        self._encoder_threads = {}
        self._saving = True
        threading.Thread(
            target=self._discard_worker,
            args=(frame_queues, encoder_threads, video_writers, video_paths),
            daemon=True,
        ).start()

    def _discard_worker(
        self,
        frame_queues: dict[str, queue.Queue],
        encoder_threads: dict[str, threading.Thread],
        video_writers: dict[str, _VideoWriter],
        video_paths: dict[str, Path],
    ) -> None:
        try:
            self._shutdown_encoders(frame_queues, encoder_threads, video_writers)
            for path in video_paths.values():
                path.unlink(missing_ok=True)
            logger.info("Recording discarded")
        except Exception:
            logger.exception("Failed to discard recording cleanly")
        finally:
            self._saving = False
            self._frame_count = 0

    def get_status(self) -> dict:
        with self._upload_lock:
            upload_status = self._upload_status
            upload_message = self._upload_message
        return {
            "recording": self._recording,
            "saving": self._saving,
            "task": self._task,
            "elapsed_s": (time.monotonic() - self._record_start_t) if self._recording else 0.0,
            "frame_count": self._frame_count,
            "episodes": list(self._episodes),
            "upload_status": upload_status,
            "upload_message": upload_message,
        }

    # --- playback ---

    def load_episode_actions(self, episode_index: int) -> list[dict] | None:
        """Returns a list of per-frame action dicts (joint/vel-name keyed, matching the
        `action` dicts RobotBridge already sends to send_action()) for the given episode.
        None if it doesn't exist or fails to load. Video is never read back by this module --
        only the recorded action column is needed for playback."""
        if episode_index < 0 or episode_index >= len(self._episodes):
            return None
        try:
            # Read the action names from info.json rather than relying on
            # configure_features() having been called on this instance -- playback should
            # work independent of the recording-setup path.
            info = json.loads((self.root / "meta" / "info.json").read_text())
            names = info["features"]["action"]["names"]
            data_path = self.root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
            df = pd.read_parquet(data_path, columns=["action"])
            return [{name: float(v) for name, v in zip(names, row)} for row in df["action"]]
        except Exception:
            logger.exception("Failed to load episode %d for playback", episode_index)
            return None

    # --- Hub upload ---

    def upload_to_hub(self) -> None:
        if self._recording or self._saving:
            logger.warning("Refusing to upload while a recording is in progress or still saving")
            return
        if not self._episodes:
            logger.warning("Nothing to upload -- no episodes recorded yet")
            return
        with self._upload_lock:
            if self._upload_status == "uploading":
                return
            self._upload_status = "uploading"
            self._upload_message = ""
        threading.Thread(target=self._upload_worker, daemon=True).start()

    def _upload_worker(self) -> None:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(repo_id=self.repo_id, folder_path=str(self.root), repo_type="dataset")
            with self._upload_lock:
                self._upload_status = "success"
                self._upload_message = f"Uploaded {len(self._episodes)} episode(s) to {self.repo_id}"
            logger.info("Upload to hub succeeded: %s", self.repo_id)
        except Exception as e:
            with self._upload_lock:
                self._upload_status = "error"
                self._upload_message = f"Upload failed: {e}"
            logger.exception("Upload to hub failed")
