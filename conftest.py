"""Project-root pytest configuration: path setup + single Quadrants init."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from skyhook._runtime import ensure_init

ensure_init()
