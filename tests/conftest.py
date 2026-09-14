from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEEPSTACK_SRC = ROOT / "src" / "deepstack"
TILESIGHT_SRC = ROOT / "src" / "tilesight"
sys.path[:0] = [str(DEEPSTACK_SRC), str(TILESIGHT_SRC)]
