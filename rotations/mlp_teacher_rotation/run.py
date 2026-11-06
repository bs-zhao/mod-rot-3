"""Run ONLY this independent mlp_teacher_rotation suite."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"np"/"src"))
from rotations.mlp_teacher_rotation.runner import main
if __name__=="__main__":
    main()
