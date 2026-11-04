"""Model generation, original prechecks, NP training, extraction and PCA/Ridge."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from modrot3.pipeline import main
if __name__ == '__main__':
    main()