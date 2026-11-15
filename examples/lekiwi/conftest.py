# The tests here (gemma_brain/tests, web_server) import the shared packages (gemma_bot.models,
# gemma_bot.protocol: packages/) from the source tree, so they run in LeRobot's environment without
# installing them. The programs themselves import the installed packages (packages/*/pyproject.toml).
import sys
from pathlib import Path

PACKAGES = Path(__file__).resolve().parent / "packages"
sys.path[:0] = [str(PACKAGES / "models" / "src"), str(PACKAGES / "protocol" / "src")]
