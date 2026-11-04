"""Plugin-based post-hoc rotations; never retrains or modifies NP."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from modrot3.rotation_engine import main
if __name__ == '__main__':
    main()