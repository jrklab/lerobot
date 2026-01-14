# ruff: noqa: E722
# Inspired by https://github.com/RoboTeacher/Pi-To-Pi-Audio-Streaming
# TODO(Yann) add license from repo above
import logging
import socket
import threading

import numpy as np

# TODO(hzy) Add sounddevice to requirements
# You might need to install sounddevice: pip install sounddevice
import sounddevice as sd

logging.basicConfig(level=logging.INFO)


class AudioClient:
    """
    Handles audio streaming to and from a remote robot. It connects to the robot's
    microphone and speaker ports to enable two-way audio communication.
    """

    def __init__(
        self,
        robot_ip: str,
        mic_port: int = 6002,
        speaker_port: int = 6003,
        sample_rate: int = 44100,
        channels: int = 1,
        format: str = "float32",
        chunk_size: int = 1024,
    ):
        """
        Initializes the AudioClient.
        Args:
            robot_ip: The IP address of the robot host.
            mic_port: The port on the robot for the microphone stream.
            speaker_port: The port on the robot for the speaker stream.
            sample_rate: Audio sample rate.
            channels: Number of audio channels.
            format: Audio format.
            chunk_size: Size of audio chunks to stream.
        """
        self.robot_ip = robot_ip
        self.mic_port = mic_port
        self.speaker_port = speaker_port
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.chunk_size = chunk_size
        self.dtype = np.dtype(self.format)

        self._running = False
        self.mic_thread = threading.Thread(target=self._mic_stream_thread, daemon=True)
        self.speaker_thread = threading.Thread(target=self._speaker_stream_thread, daemon=True)

    def start(self):
        """Starts the audio streaming threads."""
        self._running = True
        self.mic_thread.start()
        self.speaker_thread.start()
        logging.info("Audio client started.")

    def stop(self):
        """Stops the audio streaming threads."""
        self._running = False
        # The threads will exit their loops. Joining them to ensure clean shutdown.
        self.mic_thread.join(timeout=2)
        self.speaker_thread.join(timeout=2)
        logging.info("Audio client stopped.")

    def _mic_stream_thread(self):
        """Thread for receiving microphone data from the robot and playing it locally."""
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.robot_ip, self.mic_port))
                    logging.info(f"Connected to robot microphone stream at {self.robot_ip}:{self.mic_port}")
                    with sd.OutputStream(
                        samplerate=self.sample_rate, channels=self.channels, dtype=self.format
                    ) as stream:
                        bytes_per_chunk = self.chunk_size * self.dtype.itemsize * self.channels
                        while self._running:
                            data_bytes = s.recv(bytes_per_chunk)
                            if not data_bytes:
                                logging.info("Robot microphone stream closed.")
                                break
                            data_np = np.frombuffer(data_bytes, dtype=self.dtype).reshape(
                                (-1, self.channels)
                            )
                            stream.write(data_np)
            except ConnectionRefusedError:
                if self._running:
                    logging.error("Connection to robot microphone refused. Is the host running?")
            except Exception as e:
                if self._running:
                    logging.error(f"Error in mic client stream: {e}")
            finally:
                if self._running:
                    logging.info("Retrying connection to microphone stream in 5 seconds...")
                    threading.Event().wait(5)  # Use Event for stoppable wait

    def _speaker_stream_thread(self):
        """Thread for capturing local microphone audio and sending it to the robot's speaker."""
        while self._running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect((self.robot_ip, self.speaker_port))
                    logging.info(f"Connected to robot speaker stream at {self.robot_ip}:{self.speaker_port}")
                    with sd.InputStream(
                        samplerate=self.sample_rate, channels=self.channels, dtype=self.format
                    ) as stream:
                        while self._running:
                            data, overflowed = stream.read(self.chunk_size)
                            if overflowed:
                                logging.warning("Local microphone input overflowed.")
                            s.sendall(data.tobytes())
            except ConnectionRefusedError:
                if self._running:
                    logging.error("Connection to robot speaker refused. Is the host running?")
            except Exception as e:
                if self._running:
                    logging.error(f"Error in speaker client stream: {e}")
            finally:
                if self._running:
                    logging.info("Retrying connection to speaker stream in 5 seconds...")
                    threading.Event().wait(5)