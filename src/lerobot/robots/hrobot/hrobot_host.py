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

import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import draccus
import zmq

from lerobot.robots.hrobot.config_hrobot import HRobotConfig, HRobotHostConfig
from lerobot.robots.hrobot.hrobot import HRobot


@dataclass
class HRobotServerConfig:
    """Configuration for the HRobot host script."""

    robot: HRobotConfig = field(default_factory=HRobotConfig)
    host: HRobotHostConfig = field(default_factory=HRobotHostConfig)


class HRobotHost:
    def __init__(self, config: HRobotHostConfig):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

    def disconnect(self):
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: HRobotServerConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info("Configuring HRobot")
    robot = HRobot(cfg.robot)

    logging.info("Connecting HRobot")
    robot.connect()

    logging.info("Starting HostAgent")
    host = HRobotHost(cfg.host)

    last_cmd_time = time.time()
    watchdog_active = False
    logging.info("Waiting for commands...")
    try:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < host.connection_time_s:
            loop_start_time = time.time()
            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = json.loads(msg)
                robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active and (time.time() - last_cmd_time > host.watchdog_timeout_ms / 1000):
                    logging.warning(f"No command received for >{host.watchdog_timeout_ms}ms. Stopping base.")
                    robot.stop_base()
                    watchdog_active = True
            except Exception as e:
                logging.error(f"Error processing command: {e}")

            last_observation = robot.get_observation()

            for cam_key in robot.cameras:
                ret, buffer = cv2.imencode(".jpg", last_observation[cam_key], [cv2.IMWRITE_JPEG_QUALITY, 90])
                last_observation[cam_key] = base64.b64encode(buffer).decode("utf-8") if ret else ""

            try:
                host.zmq_observation_socket.send_string(json.dumps(last_observation), flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.info("Dropping observation, no client connected or client not ready.")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
    finally:
        print("Shutting down HRobot Host.")
        robot.disconnect()
        host.disconnect()
        print("HRobot Host shut down cleanly.")


if __name__ == "__main__":
    main()