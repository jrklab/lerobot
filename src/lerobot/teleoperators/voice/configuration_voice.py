from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@dataclass
class VoiceTeleopConfig(TeleoperatorConfig):
    """
    Configuration for the VoiceTeleop teleoperator.
    """
    name: str = "voice"
    # path to the vosk model relative to the project root
    model_path: str = "src/lerobot/voice_control/vosk-model-small-en-us-0.15"
