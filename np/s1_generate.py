from pathlib import Path
import json
import sys
# Support direct invocation as: python np/sN_*.py ...
NP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NP_ROOT / 'src'))
"""S1: Generate binary-choice data with the exact old C1/C2/C3/CR3 generators."""
from modrot3.cli import parse_args
from modrot3.pipeline import (dataset_path, _run_legacy, _cr3_refs, _cr3_bank,
                              _cr2_bank, _cr4_bank, _cr4_ref, _cr5_bank, _cr5_ref)
from modlearn_rot.generators.registry import get_model
from modlearn_rot.generators.dataset import generate_dataset, save_dataset, summarize_dataset
from modlearn_rot.task_specs import get_task_spec_for_model
from modlearn_rot.utils.io import save_json

def generate(args):
    target = dataset_path(args.model)
    if target.exists():
        if args.model == 'CR4':
            manifest_path = target.with_name('s1_generate__manifest.json')
            manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
            if manifest.get('x_sampling_policy') != 'iid_subject_trial_uniform_v1':
                raise RuntimeError('Legacy CR4 fixed-X dataset found. Archive outputs/choice/CR4 '
                                   'and regenerate before using IID-X CR4.')
        if args.overwrite:
            raise RuntimeError('Never overwrite a generated dataset in place; archive its output directory first.')
        print('Existing dataset preserved:', target)
        return
    if args.model == 'CR3':
        cr2, cr3 = _cr3_refs()
        # CR3 uses the original CR2-4P Fisher-optimized 500-item bank.
        if not _cr2_bank().is_file():
            _run_legacy(cr2, ['--mode', 'design', '--variants', '4',
                              '--seed', str(args.design_seed)])
        if not _cr3_bank().is_file():
            _run_legacy(cr3, ['--stage', 'design', '--seed', str(args.design_seed)])
        _run_legacy(cr3, ['--stage', 'dataset', '--subjects', str(args.subjects),
                          '--data-seed', str(args.data_seed)])
        return
    if args.model == 'CR4':
        # Fresh independent X for each subject and trial; no S0 bank.
        _run_legacy(_cr4_ref(), ['--stage', 'dataset',
                                 '--subjects', str(args.subjects),
                                 '--data-seed', str(args.data_seed)])
        return
    if args.model == 'CR5':
        # Strict controlled contrast: exactly CR3's existing bank and trial
        # order, with CR4's 3P alpha=1 choice function/prior. Never
        # regenerate or modify the reference CR3 bank here.
        if not _cr3_bank().is_file():
            raise FileNotFoundError(
                f'CR5 needs existing CR3 frozen S0 bank: {_cr3_bank()}')
        if not _cr5_bank().is_file():
            _run_legacy(_cr5_ref(), ['--stage','design'])
        _run_legacy(_cr5_ref(), ['--stage','dataset',
                                 '--subjects',str(args.subjects),
                                 '--data-seed',str(args.data_seed)])
        return
    model = get_model(args.model)
    data = generate_dataset(model, args.data_seed)
    save_dataset(data, target)
    task = get_task_spec_for_model(model)
    save_json(dict(model=model.metadata(), generator=model.manifest_extras(),
                   task_spec=task.__dict__, data_seed=args.data_seed),
              target.with_name('s1_generate__manifest.json'))
    save_json(dict(model_id=model.model_id, response_family=model.response_family,
                   has_x=True, module=task.module, **summarize_dataset(data)),
              target.with_name('s1_generate__summary.json'))
    print(f'Generated original {args.model} binary-choice data:', target)

if __name__ == '__main__':
    generate(parse_args('S1: original binary-choice data generation'))