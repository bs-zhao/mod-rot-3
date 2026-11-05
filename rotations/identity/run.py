"""Run ONLY the identity rotation suite, never other suites."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"np"/"src"))
from rotations.identity.experiment import main
if __name__=="__main__":
    main()
