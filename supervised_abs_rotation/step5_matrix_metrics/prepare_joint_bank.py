import argparse
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank


def main():
    p = argparse.ArgumentParser(description="Prepare observed-data joint polynomial/Gram bank")
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--bank-name", default="joint_bank_seed391")
    p.add_argument("--device", default="auto", choices=("auto", "numpy", "cpu", "cuda"))
    p.add_argument("--degrees", default="2,3,4")
    args = p.parse_args()
    degrees = tuple(map(int, args.degrees.split(',')))
    if not degrees or min(degrees) < 2 or len(set(degrees)) != len(degrees):
        p.error('Distinct degrees >=2 required')
    source = wave.io.default_step2(wave.ROOT, args.model)
    with threadpool_limits(limits=4):
        joint_bank.prepare(source, source.parent / "step5_matrix_metrics" / args.bank_name,
                           device=args.device, degrees=degrees)


if __name__ == "__main__":
    main()
