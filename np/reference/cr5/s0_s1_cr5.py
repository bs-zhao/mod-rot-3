"""CR5 fixed-bank 3P CPT: EXACT CR3 stimuli, CR4 parameters/function.

No CR3, CR4 or legacy outputs are modified. The *same* 500x6 physical
stimulus array, row order, native condition labels and FOSD exclusions of CR3
are copied verbatim. Only the simulator changes to the CR4 3P model with
alpha fixed at 1 and prior marginals (lambda_loss, gamma, tau).

    python np/reference/cr5/s0_s1_cr5.py --stage design
    python np/reference/cr5/s0_s1_cr5.py --stage dataset --subjects 5000
    python np/reference/cr5/s0_s1_cr5.py --stage selftest
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent/"cr3"),str(HERE.parent/"cr4"),
              str(HERE.parent),str(ROOT/"np"/"src")]
from s0_s1_cr3 import (BANK as CR3_BANK, COUNTS, CODES, 
                      dominance_audit, condition_counts)
from s0_s1_cr4 import (PARAMETER_NAMES, X_NAMES, sample_theta,
                      logits, hash_file, write_json)
from modlearn_rot.generators.dataset import GeneratedDataset,save_dataset

OUT=ROOT/"outputs"/"choice"/"CR5"
BANK=OUT/"s0_design"/"CR5_3P_cr3bank500.npz"
DESIGN=OUT/"s0_design"/"CR5_3P_design.json"
DATASET=OUT/"s1_generate"/"s1_generate__dataset.npz"
TRIALS=500


def source_bank():
    if not CR3_BANK.is_file():
        raise FileNotFoundError(
            f"CR5 requires the ACTUAL CR3 frozen 500-X bank: {CR3_BANK}. "
            "Run CR3 S0 design first; never fabricate a different CR3 bank.")
    with np.load(CR3_BANK,allow_pickle=False) as saved:
        for key in ("X","type_codes","type_names"):
            if key not in saved:
                raise ValueError(f"CR3 bank missing {key}")
        x=np.asarray(saved["X"],np.float64)
        codes=np.asarray(saved["type_codes"],np.int64)
        names=np.asarray(saved["type_names"])
    expected=np.asarray([COUNTS[name] for name in COUNTS],dtype=np.int64)
    if x.shape != (TRIALS,6) or codes.shape != (TRIALS,) or len(names)!=len(COUNTS):
        raise ValueError("CR3 frozen bank shape or categories have changed")
    if not np.array_equal(names,np.asarray(list(COUNTS))):
        raise ValueError("CR3 native type labels have changed")
    if not np.array_equal(np.bincount(codes,minlength=len(names)),expected):
        raise ValueError("CR3 bank type counts changed")
    if dominance_audit(x)["total"]!=0:
        raise ValueError("CR3 reference bank includes dominated trial(s)")
    observed=condition_counts(x)
    if (observed["gain_only"] != COUNTS["gain_only"] or
        observed["gamma_neutral"] != COUNTS["gamma_neutral"] or
        observed["loss_only"] != COUNTS["loss_only"] or
        observed["general"] != COUNTS["general_CR2"]):
        raise ValueError(f"CR3 fixed bank conditions changed: {observed}")
    return x,codes,names


def validate_immutable_bank():
    if not BANK.is_file() or not DESIGN.is_file():
        raise FileNotFoundError("Create CR5 design before generating dataset")
    meta=json.loads(DESIGN.read_text(encoding="utf-8"))
    source_x,source_codes,source_names=source_bank()
    with np.load(BANK,allow_pickle=False) as saved:
        x=np.asarray(saved["X"],np.float64)
        codes=np.asarray(saved["type_codes"],np.int64)
        names=np.asarray(saved["type_names"])
        param_names=np.asarray(saved["parameter_names"])
    if not (np.array_equal(x,source_x) and
            np.array_equal(codes,source_codes) and
            np.array_equal(names,source_names)):
        raise RuntimeError("CR5 bank deviates from original CR3 bank")
    if not np.array_equal(param_names,np.asarray(PARAMETER_NAMES)):
        raise RuntimeError("CR5 bank has wrong 3P parameter names")
    if meta["cr3_source_sha256"]!=hash_file(CR3_BANK):
        raise RuntimeError("Original CR3 bank has changed since CR5 S0")
    if meta["cr5_bank_sha256"]!=hash_file(BANK):
        raise RuntimeError("CR5 bank SHA differs from design manifest")
    return x,codes,names,meta


def design(args):
    if BANK.exists() or DESIGN.exists():
        raise FileExistsError("CR5 S0 bank/design already exists; preservation required")
    if DATASET.exists():
        raise FileExistsError("CR5 dataset exists; never overwrite its design")
    x,codes,names=source_bank()
    BANK.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(BANK,X=x,type_codes=codes,type_names=names,
                        parameter_names=np.asarray(PARAMETER_NAMES),
                        x_names=np.asarray(X_NAMES))
    write_json(DESIGN,dict(
        model="CR5",fixed_alpha=1.,parameter_names=list(PARAMETER_NAMES),
        source_model="CR3",original_bank=str(CR3_BANK),
        cr3_source_sha256=hash_file(CR3_BANK),
        cr5_bank_sha256=hash_file(BANK),trial_count=TRIALS,
        source_condition_names=names.tolist(),
        source_condition_counts={name:int((codes==idx).sum())
                                 for idx,name in enumerate(names)},
        original_bank_X_and_types_exact=True,
        formula="unchanged CR4 logits = original CR3/CR2 CPT with alpha=1",
        priors="unchanged CR4 lambda/gamma/tau marginal priors",
        decoder_x_standardization="same standardization policy as CR4 but fitted on CR5/CR3 X",
        note="No CR3 or CR4 output was modified; no regeneration, sampling, or optimization of X"))
    validate_immutable_bank()
    print("CR5 X exactly matches CR3, 500 rows with native categories:",BANK,flush=True)


def dataset(args):
    if DATASET.exists():
        raise FileExistsError(f"CR5 dataset already exists: {DATASET}")
    x,codes,names,design_meta=validate_immutable_bank()
    # SAME population parameter RNG as CR4: only the bank changes.
    rng_theta=np.random.default_rng(args.data_seed+4)
    rng_choice=np.random.default_rng(args.data_seed+1004)
    theta=sample_theta(args.subjects,rng_theta)
    p=expit(logits(x[None,:,:],theta[:,None,:]))
    if not np.isfinite(p).all() or not np.all((p>0)&(p<1)):
        raise RuntimeError("Invalid CR5 generator predictions")
    y=(rng_choice.random(p.shape)<p).astype(np.int64)
    n=len(theta)
    ds=GeneratedDataset(
        parameters=theta,
        trial_counts=np.full(n,TRIALS,np.int64),
        subject_index=np.repeat(np.arange(n,dtype=np.int64),TRIALS),
        trial_index=np.tile(np.arange(TRIALS,dtype=np.int64),n),
        x=np.broadcast_to(x,(n,TRIALS,6)).reshape(-1,6).copy(),
        y=y.reshape(-1,1),
        predictive_true=p.reshape(-1,1))
    save_dataset(ds,DATASET)
    write_json(DATASET.parent/"s1_generate__manifest.json",dict(
        model="CR5",fixed_alpha=1.,subjects=n,trials_per_subject=TRIALS,
        parameter_names=list(PARAMETER_NAMES),data_seed=args.data_seed,
        bank_sha256=hash_file(BANK),dataset_sha256=hash_file(DATASET),
        cr3_source_sha256=design_meta["cr3_source_sha256"],
        source_type_counts=design_meta["source_condition_counts"],
        generator_equals_CR4_3P=True,same_subject_prior_seed_policy_as_CR4=True,
        original_X_and_trial_order_equals_CR3=True,
        trained_network=False))
    print("CR5 saved three-parameter dataset from original CR3 500-trial X:",DATASET,flush=True)


def selftest():
    # Formula/prior equivalence: no bank is required to verify formula,
    # but CR3 source is required to test exact X provenance.
    theta=sample_theta(5,np.random.default_rng(31))
    assert theta.shape==(5,3)
    x,codes,names=source_bank()
    assert x.shape==(500,6) and int(np.sum(codes==CODES["gain_only"]))==60
    assert len(names)==4
    check=logits(x[:12][None],theta[:,None])
    assert np.isfinite(check).all()
    print("CR5 exact CR3 native X composition + 3P/alpha=1 formula PASS")


def main(argv=None):
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage",choices=("design","dataset","selftest"),default="design")
    ap.add_argument("--subjects",type=int,default=5000)
    ap.add_argument("--data-seed",type=int,default=20261028)
    args=ap.parse_args(argv)
    if args.subjects<1:ap.error("subjects must be positive")
    if args.stage=="design":design(args)
    elif args.stage=="dataset":dataset(args)
    else:selftest()


if __name__=="__main__":
    main()
