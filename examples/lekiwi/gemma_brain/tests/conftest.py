# The tests import gemma_brain's packages (brain/, common/, calibration/) and the Pi-side
# web_server's simulation (test_auto_control.Sim), the stand-in executor the brain is tested against.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "web_server")]
