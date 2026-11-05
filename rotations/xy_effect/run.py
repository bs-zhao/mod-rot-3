"""Run ONLY the xy_effect rotation suite, never other suites."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"np"/"src"))
from rotations.xy_effect.experiment import main
if __name__=="__main__":
    main()
