"""Launch only the functional_composition rotation suite."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'np' / 'src'))
from rotations.functional_composition.runner import main
if __name__ == '__main__':
    main()