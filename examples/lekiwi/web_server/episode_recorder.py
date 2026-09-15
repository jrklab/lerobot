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
Episode recording + playback for the LeKiwi web app, backed by `LeRobotDataset`.

Recording captures whatever the robot is doing regardless of which control mode is driving
it (gamepad/web/leader arm) -- it just observes the same obs/action dicts RobotBridge's
control loop already computes every tick. Playback takes exclusive control of the robot for
the duration of one episode's replay; that part is handled by RobotBridge itself (see its
`_control_loop()`), not here -- this module only owns the dataset lifecycle.
"""

import logging
import threading
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

logger = logging.getLogger(__name__)

# h264 instead of the framework's default libsvtav1 (AV1): AV1 software encoding is too
# CPU-heavy for real-time dual-camera capture on a Raspberry Pi 4.
VCODEC = "h264"


class EpisodeRecorder:
    """Owns one dataset's recording/playback/upload lifecycle. Not thread-safe against
    concurrent calls to its mutating methods -- RobotBridge only ever calls these from
    either its own control-loop thread (add_frame) or serialized WS-message handling on
    the FastAPI event loop (everything else), never both at once for the same call."""

    def __init__(self, repo_id: str, root: str | Path | None, fps: int):
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.fps = fps
        self._dataset_features: dict | None = None

        self._dataset: LeRobotDataset | None = None
        self._recording = False
        self._task = ""
        self._record_start_t = 0.0
        self._frame_count = 0

        self._episodes: list[dict] = []
        self._load_existing_episodes()

        # save_episode()/clear_episode_buffer() (video encoding, disk I/O) run on a
        # background thread -- stop_recording()/discard_recording() are called synchronously
        # from the FastAPI event loop via RobotBridge, and blocking that loop would freeze the
        # WS connection (state pushes, other messages) for every connected client meanwhile.
        self._saving = False

        self._upload_lock = threading.Lock()
        self._upload_status = "idle"  # "idle" | "uploading" | "success" | "error"
        self._upload_message = ""

    def configure_features(self, action_features: dict, observation_features: dict) -> None:
        """Must be called once (with the real robot's feature dicts) before recording can
        start. Deferred out of __init__ since RobotBridge only knows these after connect()."""
        self._dataset_features = {
            **hw_to_dataset_features(action_features, ACTION),
            **hw_to_dataset_features(observation_features, OBS_STR),
        }

    def _load_existing_episodes(self) -> None:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return  # Brand new dataset -- nothing to seed, and importantly, no Hub network
            # call gets triggered by probing further (LeRobotDatasetMetadata falls back to
            # downloading from the Hub if local metadata is missing, which we don't want to
            # risk here -- e.g. no internet on the Pi's hotspot-only network mode).
        try:
            import pyarrow.parquet as pq

            rows = []
            for pq_file in sorted((self.root / "meta" / "episodes").glob("*/*.parquet")):
                table = pq.read_table(pq_file, columns=["episode_index", "tasks", "length"])
                rows.extend(table.to_pylist())
            rows.sort(key=lambda r: r["episode_index"])
            self._episodes = [
                {
                    "index": r["episode_index"],
                    "task": r["tasks"][0] if r["tasks"] else "",
                    "length": r["length"],
                    "duration_s": r["length"] / self.fps,
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

    def start_recording(self, task: str) -> bool:
        if self._recording or self._saving:
            return False
        if self._dataset_features is None:
            raise RuntimeError("configure_features() must be called before recording")
        try:
            if (self.root / "meta" / "info.json").exists():
                self._dataset = LeRobotDataset.resume(
                    repo_id=self.repo_id, root=self.root, vcodec=VCODEC, streaming_encoding=True
                )
            else:
                self._dataset = LeRobotDataset.create(
                    repo_id=self.repo_id,
                    fps=self.fps,
                    features=self._dataset_features,
                    root=self.root,
                    robot_type="lekiwi",
                    use_videos=True,
                    vcodec=VCODEC,
                    streaming_encoding=True,
                )
        except Exception:
            logger.exception("Failed to open dataset for recording")
            self._dataset = None
            return False

        self._task = task
        self._record_start_t = time.monotonic()
        self._frame_count = 0
        self._recording = True
        logger.info("Recording started: task=%r", task)
        return True

    def add_frame(self, obs: dict, action: dict) -> None:
        """Called every control-loop tick while recording is active. Any failure stops the
        recording rather than crashing the caller's control loop."""
        if not self._recording or self._dataset is None:
            return
        try:
            obs_frame = build_dataset_frame(self._dataset.features, obs, OBS_STR)
            action_frame = build_dataset_frame(self._dataset.features, action, ACTION)
            self._dataset.add_frame({**obs_frame, **action_frame, "task": self._task})
            self._frame_count += 1
        except Exception:
            logger.exception("add_frame failed -- stopping recording")
            self._recording = False

    def stop_recording(self) -> None:
        if not self._recording or self._dataset is None:
            return
        dataset = self._dataset
        task = self._task
        frame_count = self._frame_count
        # Stop accepting new frames immediately (synchronous) so add_frame() calls from the
        # control-loop thread cease at once; the slow part (video encode + parquet write)
        # happens on a background thread so it never blocks the FastAPI event loop.
        self._recording = False
        self._dataset = None
        self._saving = True
        threading.Thread(
            target=self._save_worker, args=(dataset, task, frame_count), daemon=True
        ).start()

    def _save_worker(self, dataset: LeRobotDataset, task: str, frame_count: int) -> None:
        try:
            dataset.save_episode()
            dataset.finalize()
            self._episodes.append(
                {
                    "index": len(self._episodes),
                    "task": task,
                    "length": frame_count,
                    "duration_s": frame_count / self.fps,
                }
            )
            logger.info("Recording saved: episode %d, %d frames", len(self._episodes) - 1, frame_count)
        except Exception:
            logger.exception("Failed to save episode")
        finally:
            self._saving = False
            self._frame_count = 0

    def discard_recording(self) -> None:
        if not self._recording or self._dataset is None:
            return
        dataset = self._dataset
        self._recording = False
        self._dataset = None
        self._saving = True
        threading.Thread(target=self._discard_worker, args=(dataset, self._frame_count), daemon=True).start()

    def _discard_worker(self, dataset: LeRobotDataset, frame_count: int) -> None:
        try:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.info("Recording discarded (%d frames)", frame_count)
        except Exception:
            logger.exception("Failed to discard episode buffer")
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

    def load_episode_actions(self, episode_index: int) -> list[dict] | None:
        """Returns a list of per-frame action dicts (joint/vel-name keyed, matching the
        `action` dicts RobotBridge already sends to send_action()) for the given episode.
        None if it doesn't exist or fails to load."""
        if episode_index < 0 or episode_index >= len(self._episodes):
            return None
        try:
            dataset = LeRobotDataset(self.repo_id, root=self.root, episodes=[episode_index])
            names = dataset.features[ACTION]["names"]
            actions_col = dataset.select_columns(ACTION)
            return [
                {name: float(actions_col[i][ACTION][j]) for j, name in enumerate(names)}
                for i in range(dataset.num_frames)
            ]
        except Exception:
            logger.exception("Failed to load episode %d for playback", episode_index)
            return None

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
            dataset = LeRobotDataset(self.repo_id, root=self.root)
            dataset.push_to_hub()
            with self._upload_lock:
                self._upload_status = "success"
                self._upload_message = f"Uploaded {len(self._episodes)} episode(s) to {self.repo_id}"
            logger.info("Upload to hub succeeded: %s", self.repo_id)
        except Exception as e:
            with self._upload_lock:
                self._upload_status = "error"
                self._upload_message = f"Upload failed: {e}"
            logger.exception("Upload to hub failed")
