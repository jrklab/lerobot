# The tests import gemma_brain's packages (brain/, common/, calibration/) and the Pi-side
# web_server modules (auto_control's limits, its arm simulator) the client code uses.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "web_server")]
