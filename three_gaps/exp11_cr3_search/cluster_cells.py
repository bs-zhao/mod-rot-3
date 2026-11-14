"""EXP11b: loss cells without design flags.  W-only mode-flag contexts -> cluster their (trace-normalised) cell energy matrices
E_c = mean g g^T by average-linkage into K groups -> use the groups as loss cells (floor-corrected colLog3, minimax axis from the contexts).
python cluster_cells.py MODEL=CR3 REPEAT CHANNEL K"""
import sys, os, numpy as np, torch
os.environ["GENERIC_CTX"] = "1"
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import harness as H, floor as FL, minimaxA as MA
from common.rotlib import bank_categories
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
torch.set_num_threads(4)
rep, ch, K = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
cond = FL.CONDS["CR3"] + ("_rep02" if rep == 2 else "")
c = H.load("CR3", cond, "decoder", ch); d = 4
mats = MA.ctx_matrices(c, ch, "gain")
rng = np.random.default_rng(0)
e1, w1 = MA.minimax_axis(mats, d, rng); F = w1[None]
E = FL.cell_energies(c).numpy()                               # per generic context
En = E / np.trace(E, axis1=1, axis2=2)[:, None, None]
D = np.array([[np.linalg.norm(a - b) for b in En] for a in En])
Z = linkage(squareform(D, checks=False), "average"); grp = fcluster(Z, K, "maxclust") - 1
ctx = c["cat"].numpy(); des = bank_categories(c["X"]).numpy()
newcat = grp[ctx]
print(f"rep{rep} {ch} K={K}: context->cell {grp.tolist()} ; sizes {np.bincount(newcat).tolist()}")
print("designed (rows) x clustered cells (cols):\n", np.array([[((des == a) & (newcat == b)).sum() for b in range(newcat.max() + 1)] for a in range(4)]))
R = np.load(H.io.ROOT / "outputs" / "three_gaps" / "exp00" / f"CR3_{cond}_pca0" / "R_oracle_raw.npy")
c["cat"] = torch.tensor(newcat, dtype=torch.long)
for loss in ["colLog3", "quartimax"]:
    f, lam, sig2, _ = FL.make_loss(c, loss)
    fin = H.optimize(c, f, "SO", F=F, starts=12, steps=250, seed=7)
    print(f"   {loss:9s} clustered cells: TEST mcc/leak/max %.3f %.3f %.3f   (axis e1 {e1:.3f}; oracle %.3f/%.3f/%.3f)" % (H.evaluate(c, fin[0][1]) + H.evaluate(c, R)), flush=True)
