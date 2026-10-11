from pathlib import Path
import sys

# Make `backend` importable when pytest is launched from the repository root
# or from another working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
