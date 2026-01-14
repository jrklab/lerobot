# ruff: noqa: E722
# Inspired by https://github.com/RoboTeacher/Pi-To-Pi-Audio-Streaming
# TODO(Yann) add license from repo above
import logging
import socket
import threading
import time

import numpy as np

# TODO(hzy) Add sounddevice to requirements
# You might need to install sounddevice: pip install sounddevice
# You might also need to install portaudio: sudo apt-get install libportaudio2
import sounddevice as sd

logging.basicConfig(level=logging.INFO)


class AudioHandler:
    """
    Handles audio streaming from the robot's microphone to a client and from a client to the robot's speaker.
    It opens two TCP server sockets, one for the microphone stream and one for the speaker stream.
    """

    def __init__(
        self,
        audio_device: str = "USB",
        mic_port: int = 6002,
        speaker_port: int = 6003,
        sample_rate: int = 44100,
        channels: int = 1,
        format: str = "float32",
        chunk_size: int = 1024,
    ):
        """
        Initializes the AudioHandler.
        Args:
            audio_device: Substring of the audio device name to use (e.g., "USB").
            mic_port: Port for the microphone stream.
            speaker_port: Port for the speaker stream.
            sample_rate: Audio sample rate.
            channels: Number of audio channels.
            format: Audio format.
            chunk_size: Size of audio chunks to stream.
        """
        self.mic_port = mic_port
        self.speaker_port = speaker_port
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.chunk_size = chunk_size
        self.dtype = np.dtype(self.format)

        self.device_id = self._find_device_id(audio_device)
        if self.device_id is None:
            raise RuntimeError(f"Could not find an audio device with name containing '{audio_device}'")

        self._running = False
        self.mic_thread = threading.Thread(target=self._mic_stream_thread, daemon=True)
        self.speaker_thread = threading.Thread(target=self._speaker_stream_thread, daemon=True)

    def _find_device_id(self, device_name_substring: str):
        """Finds the device ID for a given device name substring."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name_substring in device["name"]:
                logging.info(f"Found audio device '{device['name']}' with ID {i}")
                return i
        logging.error(f"Could not find audio device containing '{device_name_substring}'.")
        logging.info("Available devices:")
        for i, device in enumerate(devices):
            logging.info(f"  {i}: {device['name']}")
        return None

    def start(self):
        """Starts the audio streaming threads."""
        self._running = True
        self.mic_thread.start()
        self.speaker_thread.start()
        logging.info("Audio handler started.")

    def stop(self):
        """Stops the audio streaming threads."""
        self._running = False
        # Create dummy connections to unblock the accept() calls
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", self.mic_port))
        except:
            pass
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", self.speaker_port))
        except:
            pass
        self.mic_thread.join(timeout=2)
        self.speaker_thread.join(timeout=2)
        logging.info("Audio handler stopped.")

    def _mic_stream_thread(self):
        """Thread for streaming microphone data."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(("", self.mic_port))
            server_socket.listen(1)
            logging.info(f"Microphone stream listening on port {self.mic_port}")

            while self._running:
                conn, addr = server_socket.accept()
                if not self._running:
                    break
                with conn:
                    logging.info(f"Microphone client connected: {addr}")
                    try:
                        with sd.InputStream(
                            samplerate=self.sample_rate,
                            device=self.device_id,
                            channels=self.channels,
                            dtype=self.format,
                        ) as stream:
                            while self._running:
                                data, overflowed = stream.read(self.chunk_size)
                                if overflowed:
                                    logging.warning("Microphone input overflowed")
                                conn.sendall(data.tobytes())
                    except Exception as e:
                        logging.error(f"Error in mic stream: {e}")
                    finally:
                        logging.info(f"Microphone client disconnected: {addr}")

    def _speaker_stream_thread(self):
        """Thread for receiving and playing speaker data."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(("", self.speaker_port))
            server_socket.listen(1)
            logging.info(f"Speaker stream listening on port {self.speaker_port}")

            while self._running:
                conn, addr = server_socket.accept()
                if not self._running:
                    break
                with conn:
                    logging.info(f"Speaker client connected: {addr}")
                    try:
                        with sd.OutputStream(
                            samplerate=self.sample_rate,
                            device=self.device_id,
                            channels=self.channels,
                            dtype=self.format,
                        ) as stream:
                            bytes_per_chunk = self.chunk_size * self.dtype.itemsize * self.channels
                            while self._running:
                                data_bytes = conn.recv(bytes_per_chunk)
                                if not data_bytes:
                                    break
                                data_np = np.frombuffer(data_bytes, dtype=self.dtype).reshape(
                                    (-1, self.channels)
                                )
                                stream.write(data_np)
                    except Exception as e:
                        logging.error(f"Error in speaker stream: {e}")
                    finally:
                        logging.info(f"Speaker client disconnected: {addr}")


if __name__ == "__main__":
    # Example usage:
    # This can be run on the robot to test the audio handler independently.
    print("Starting audio handler...")
    handler = AudioHandler()
    handler.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping audio handler...")
        handler.stop()
        print("Audio handler stopped.")
