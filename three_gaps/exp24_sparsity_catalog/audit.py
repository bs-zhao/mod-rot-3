"""EXP24 audit of the measure catalogue on prescribed example vectors (no data).

Checks, for every entry-level measure (lower = sparser):
  * the user's example [3,2,.01,.01] vs [1,1,.5,.1]: must call the first one sparser (interpreted as energies and as amplitudes);
  * Hoyer-type pairs [2,.011,.001] vs [2,.001,.001] (tail sensitivity), [1,0,0,0] vs [1,1,1,1] (extreme), Robin-Hood-type transfer;
  * invariance to global rescaling and to permutation.
python audit.py > results/audit.md
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import catalog as C

torch.set_default_dtype(torch.float64)
V = lambda *a: torch.tensor(a, dtype=torch.float64)
PAIRS = {  # name: (should-be-sparser, other), vectors are AMPLITUDES unless suffixed _e
    "user [3,2,.01,.01] < [1,1,.5,.1]": (V(3, 2, .01, .01), V(1, 1, .5, .1)),
    "tail [2,.001,.001] < [2,.011,.001]": (V(2, .001, .001), V(2, .011, .001)),
    "extreme [1,0,0,0] < [1,1,1,1]": (V(1, 0, 0, 0), V(1, 1, 1, 1)),
    "concentration [4,1,1,1] < [2,2,2,1]": (V(4, 1, 1, 1), V(2, 2, 2, 1)),
}
ENTRY = {k: v for k, v in C.UX.items() if k not in C.STRUCTURED}
rows = []
for name, f in ENTRY.items():
    r = dict(measure=name)
    for pname, (a, b) in PAIRS.items():
        for interp, fn in (("amp", lambda x: x ** 2), ("energy", lambda x: x)):
            la, lb = float(f(fn(a))), float(f(fn(b)))
            r[f"{pname.split(' ')[0]}_{interp}"] = la < lb - 1e-9
    e = V(3, 2, .01, .01) ** 2
    r["scale_invariant"] = abs(float(f(e)) - float(f(e * 37.0))) < 1e-6 * (1 + abs(float(f(e))))
    r["perm_invariant"] = abs(float(f(e)) - float(f(e[torch.tensor([2, 0, 3, 1])]))) < 1e-9
    rows.append(r)
df = pd.DataFrame(rows).set_index("measure")
df["passes_user_example"] = df["user_amp"] & df["user_energy"]
print("# EXP24 audit (entry-level measures; True = first vector judged sparser / property holds)\n")
print(df.to_markdown())
print("\nMeasures failing the user example (either interpretation):", list(df.index[~df.passes_user_example]))
df.to_csv(Path(__file__).resolve().parent / "results" / "audit.csv")
