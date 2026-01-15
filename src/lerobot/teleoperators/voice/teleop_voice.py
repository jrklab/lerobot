import json
import logging
import os
import queue
import sys
import threading
from typing import Any


from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..teleoperator import Teleoperator
from .configuration_voice import VoiceTeleopConfig

VOICE_AVAILABLE = True
try:
    import sounddevice as sd
    from vosk import KaldiRecognizer, Model
    from .command_parser import CommandParser
except ImportError:
    VOICE_AVAILABLE = False


class VoiceTeleop(Teleoperator):
    """
    Teleop class to use voice inputs for control.
    """

    config_class = VoiceTeleopConfig
    name = "voice"

    def __init__(self, config: VoiceTeleopConfig):
        super().__init__(config)
        self.config = config
        self.command_queue = queue.Queue()
        self.listener_thread = None
        self._stop_event = threading.Event()

        if not VOICE_AVAILABLE:
            raise ImportError("Voice dependencies are not installed. Please install sounddevice, vosk.")

        self.model = None
        self.recognizer = None
        self.parser = CommandParser()
        self._is_connected = False


    def _initialize_recognizer(self):
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            # try relative from the voice teleoperator directory
            model_path = os.path.join("src", "lerobot", "teleoperators", "voice", os.path.basename(model_path))
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Vosk model not found at '{self.config.model_path}' or '{model_path}'.")

        self.model = Model(model_path)
        
        try:
            device_info = sd.query_devices(None, 'input')
            samplerate = int(device_info['default_samplerate'])
        except Exception as e:
            logging.warning(f"Could not get default device sample rate: {e}. Falling back to 44100 Hz.")
            samplerate = 44100
            
        self.recognizer = KaldiRecognizer(self.model, samplerate)
        self.recognizer.SetWords(True)
        return samplerate

    def _voice_recognition_loop(self, samplerate):
        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            audio_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None, dtype='int16',
                                    channels=1, callback=audio_callback):
                logging.info("Voice teleop listening...")
                while not self._stop_event.is_set():
                    data = audio_queue.get(timeout=1)
                    if self.recognizer.AcceptWaveform(data):
                        result_json = self.recognizer.Result()
                        result = json.loads(result_json)
                        text = result.get('text', '')
                        
                        if text:
                            command = self.parser.parse_command(text)
                            if command:
                                self.command_queue.put(command)
        except queue.Empty:
            # this is expected when no audio is coming in
            pass
        except Exception as e:
            logging.error(f"Voice recognition error: {e}")
        finally:
            logging.info("Voice teleop stopped listening.")

    @property
    def action_features(self) -> dict:
        return {"voice_command": (str, str)}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("VoiceTeleop is already connected.")

        samplerate = self._initialize_recognizer()
        
        self.listener_thread = threading.Thread(target=self._voice_recognition_loop, args=(samplerate,))
        self.listener_thread.daemon = True
        self.listener_thread.start()
        self._is_connected = True
        logging.info("VoiceTeleop connected.")

    @property
    def is_calibrated(self) -> bool:
        return True # no calibration needed

    def calibrate(self) -> None:
        pass # no calibration needed

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        try:
            command = self.command_queue.get_nowait()
            return {"voice_command": command}
        except queue.Empty:
            return {"voice_command": None}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass # No feedback to send

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError("VoiceTeleop is not connected.")
        
        self._stop_event.set()
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        
        self._is_connected = False
        logging.info("VoiceTeleop disconnected.")
