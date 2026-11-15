# The tests here (gemma_brain/tests, web_server) import the shared robot models (gemma_bot.models,
# packages/models) from the source tree, so they run in LeRobot's environment without installing
# it. The programs themselves import the installed package (packages/models/pyproject.toml).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "packages" / "models" / "src"))
