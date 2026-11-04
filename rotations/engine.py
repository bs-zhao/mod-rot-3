"""Scalable method-agnostic rotation runner.

Each independent folder rotations/<method>/ defines NAME, VERSION, DEFAULTS,
and either objective(data, R, config) or fit(data, config). The engine never
edits its method registry to add an algorithm. All methods are fit on frozen
ZXY TRAIN participants; true parameter values are loaded only after saving R.
"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

# Method discovery must remain independent of the baseline NN/simulator imports.
ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
from .api import RotationInput, RotationResult, optimize_objective, validate_result
from .data import aligned_observations, evaluate_after_freeze

BUILTINS = ROOT / 'rotations'
ALLOWED_NAME = re.compile(r'^[a-z][a-z0-9_]*$')


@dataclass(frozen=True)
class Plugin:
    name: str
    version: str
    description: str
    defaults: dict[str, Any]
    module: Any
    path: Path
    code_sha256: str


def _method_name_from_path(path: Path) -> str:
    return path.parent.name if path.name == '__init__.py' else path.stem


def method_code_sha(path: Path) -> str:
    """Hash all Python files of an algorithm package, not just its entrypoint."""
    if path.name != '__init__.py':
        return sha256(path)
    paths = sorted(path.parent.rglob('*.py'))
    h = hashlib.sha256()
    for item in paths:
        h.update(item.relative_to(path.parent).as_posix().encode('utf-8'))
        h.update(item.read_bytes())
    return h.hexdigest()


def discover_paths(directories: list[Path]) -> dict[str, Path]:
    """Cheap discovery: does not import hundreds of optional plugin dependencies.

    One-file methods and multi-file packages are both supported. A package
    needs a `name/__init__.py`, and may import relative helper modules.
    """
    paths: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f'Plugin directory not found: {directory}')
        candidates = (sorted(directory.glob('*/__init__.py')) if directory.resolve() == BUILTINS.resolve()
                      else sorted(directory.glob('*.py')) + sorted(directory.glob('*/__init__.py')))
        for path in candidates:
            name = _method_name_from_path(path)
            if path.name == '__init__.py' and path.parent == directory:
                continue  # directory package marker, not a method
            if name.startswith('_'):
                continue
            if not ALLOWED_NAME.fullmatch(name):
                raise ValueError(f'Invalid plugin name {name!r}')
            if name in paths:
                raise ValueError(f'Duplicate method name {name!r}: {paths[name]}, {path}')
            paths[name] = path
    return paths


def load_plugin(path: Path) -> Plugin:
    name = _method_name_from_path(path)
    if not ALLOWED_NAME.fullmatch(name):
        raise ValueError(f'Unsafe method module name {path}')
    content_sha = method_code_sha(path)
    qualified = f'_modrot3_method_{name}_{content_sha[:12]}'
    is_package = path.name == '__init__.py'
    spec = importlib.util.spec_from_file_location(
        qualified, path, submodule_search_locations=[str(path.parent)] if is_package else None)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import method file {path}')
    module = importlib.util.module_from_spec(spec)
    # A Python package must exist in sys.modules for relative imports.
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified, None)
        raise
    declared = getattr(module, 'NAME', None)
    if declared != name:
        raise ValueError(f'{path}: NAME must equal lowercase module/package name {name!r}')
    version = getattr(module, 'VERSION', None)
    if not isinstance(version, str) or not version:
        raise ValueError(f'{path}: VERSION is required')
    defaults = getattr(module, 'DEFAULTS', None)
    if not isinstance(defaults, dict) or not all(isinstance(k, str) for k in defaults):
        raise ValueError(f'{path}: DEFAULTS must be a str->value dict')
    if not callable(getattr(module, 'fit', None)) and not callable(getattr(module, 'objective', None)):
        raise ValueError(f'{path}: supply fit(data,config) or objective(data,R,config)')
    return Plugin(name=name, version=version,
                  description=str(getattr(module, 'DESCRIPTION', '')),
                  defaults=defaults.copy(), module=module, path=path, code_sha256=content_sha)


def discover(directories: list[Path]):
    """Eager loader for inspection/testing; execution uses lazy discover_paths."""
    plugins = {}
    failures = {}
    for name, path in discover_paths(directories).items():
        try:
            plugins[name] = load_plugin(path)
        except Exception as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
    return plugins, failures


def inspect_metadata(path: Path):
    """Read constant NAME/VERSION/DESCRIPTION without executing optional deps."""
    values = {}
    try:
        t = ast.parse(path.read_text(encoding='utf-8'))
        for node in t.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('NAME', 'VERSION', 'DESCRIPTION')):
                try:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    except (OSError, SyntaxError):
        pass
    return values

def _value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_overrides(values: list[str]):
    global_set = {}
    per_method = {}
    for item in values:
        if '=' not in item:
            raise ValueError(f'Expected --set parameter=value, got {item!r}')
        key, raw = item.split('=', 1)
        val = _value(raw)
        if '.' in key:
            method, param = key.split('.', 1)
            per_method.setdefault(method, {})[param] = val
        else:
            global_set[key] = val
    return global_set, per_method


def _read_plan(path: Path | None, methods: list[str]):
    if path is None:
        if not methods:
            return [{'method': 'identity'}]
        return [{'method': name} for name in methods]
    plan = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(plan, list) or not plan:
        raise ValueError('Plan JSON must be a nonempty array')
    for item in plan:
        if not isinstance(item, dict) or not isinstance(item.get('method'), str):
            raise ValueError('Each plan entry needs {"method": "..."}')
        if not isinstance(item.get('config', {}), dict):
            raise ValueError('Plan "config" must be a dict')
    if methods:
        raise ValueError('--plan may not be mixed with --method/--methods')
    return plan


def _configs(plugin: Plugin, args, item: dict):
    cfg = dict(plugin.defaults)
    # Per-method defaults live beside each method, independent of the runner.
    local_config = plugin.path.parent / 'config.json' if plugin.path.name == '__init__.py' else None
    if local_config is not None and local_config.is_file():
        values = json.loads(local_config.read_text(encoding='utf-8'))
        if not isinstance(values, dict) or set(values) - set(cfg):
            raise ValueError(f'{plugin.name}: invalid local config.json keys')
        cfg.update(values)
    if 'seed' in cfg:
        cfg['seed'] = args.seed
    if args.starts is not None and 'starts' in cfg:
        cfg['starts'] = args.starts
    if args.maxiter is not None and 'maxiter' in cfg:
        cfg['maxiter'] = args.maxiter
    cfg_file = {'defaults': {}, 'methods': {}}
    if args.config_file:
        cfg_file = json.loads(Path(args.config_file).read_text(encoding='utf-8'))
        if not isinstance(cfg_file, dict) or not isinstance(cfg_file.get('defaults', {}), dict) or not isinstance(cfg_file.get('methods', {}), dict):
            raise ValueError('Config JSON needs optional defaults/methods dicts')
    override_global, override_methods = _set_overrides(args.set)
    for vals in (cfg_file.get('defaults', {}), cfg_file.get('methods', {}).get(plugin.name, {}),
                 override_global, override_methods.get(plugin.name, {}), item.get('config', {})):
        if not isinstance(vals, dict):
            raise ValueError('Method configuration overrides must be dicts')
        unexpected = set(vals) - set(plugin.defaults)
        if unexpected:
            raise ValueError(f'{plugin.name}: unknown config keys {sorted(unexpected)}; allowed: {sorted(plugin.defaults)}')
        cfg.update(vals)
    return cfg


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (int, float, bool, str, type(None))):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    raise TypeError(f'Cannot JSON serialize {type(x)!r}')


def _digest_json(payload):
    payload = json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def _build_source(args):
    from modrot3.pipeline import dataset_path
    from modlearn_rot.generators.dataset import load_dataset
    if args.model not in ('C1', 'C2', 'C3', 'CR3'):
        raise ValueError('Supported models: C1 C2 C3 CR3')
    root = ROOT / 'outputs' / 'choice' / args.model
    src = root/'s4_evaluate'/args.condition/f'repeat{args.repeat:02d}'/'s4_evaluate__latent.npz'
    if not src.exists():
        raise FileNotFoundError(f'Extract frozen Z first: {src}')
    observed = dataset_path(args.model)
    if not observed.exists():
        raise FileNotFoundError(f'Generate original X,Y first: {observed}')
    z_sha, d_sha = sha256(src), sha256(observed)
    with np.load(src, allow_pickle=False) as f:
        if str(f['context_mode']) != 'all_observed_trials':
            raise RuntimeError('Rotation requires posterior means encoded using ALL observed trials')
        if str(f['dataset_sha256']) != d_sha:
            raise RuntimeError('Z was extracted from another dataset; refusing to mix X/Y')
        z = np.array(f['z'], float)
        tr = np.array(f['train_ids'], int)
        val = np.array(f['validation_ids'], int)
        test = np.array(f['test_ids'], int)
        checkpoint_sha = str(f['checkpoint_sha256'])
    if len(np.unique(np.r_[tr,val,test])) != len(z) or not np.array_equal(
        np.sort(np.r_[tr,val,test]), np.arange(len(z))):
        raise ValueError('S4 subject splits do not partition all participants')
    native_k = z.shape[1]
    if args.representation == 'full':
        k = args.rotation_dim or native_k
        if k != native_k:
            raise ValueError('--representation full requires --rotation-dim equal to latent dimension')
    else:
        k = args.rotation_dim or {'C1': 4, 'C2': 4, 'C3': 6, 'CR3': 4}[args.model]
    if k < 2 or k > native_k:
        raise ValueError('rotation-dim must be between 2 and the native Z dimension')
    if args.fit_max_subjects < 1:
        raise ValueError('--fit-max-subjects must be >=1')
    pca = PCA(n_components=k, svd_solver='full').fit(z[tr])
    floor = args.whiten_floor
    if floor <= 0:
        raise ValueError('whiten-floor must be strictly positive')
    W = pca.components_ / np.sqrt(np.maximum(pca.explained_variance_, floor))[:,None]
    q = (z - pca.mean_) @ W.T
    chosen = np.sort(np.random.default_rng(args.seed).choice(tr, min(len(tr), args.fit_max_subjects), replace=False))
    generated = load_dataset(observed)
    # Data parameters are deliberately NOT used or passed to a plugin here.
    X, Y, mask = aligned_observations(generated)
    if len(z) != len(X):
        raise RuntimeError('Latent rows and observed subject rows do not match')
    data = RotationInput(q=q[chosen], x=X[chosen], y=Y[chosen], mask=mask[chosen])
    return dict(z=z, tr=tr, val=val, test=test, chosen=chosen, pca=pca, W=W,
                data=data, latent_sha=z_sha, dataset_sha=d_sha, checkpoint_sha=checkpoint_sha,
                root=root, dataset=generated)


def _invoke(plugin: Plugin, data: RotationInput, config: dict[str, Any]):
    if callable(getattr(plugin.module, 'fit', None)):
        result = plugin.module.fit(data, config)
    else:
        result = optimize_objective(data, plugin.module.objective, config)
    return validate_result(result, data.k)


def _run_one(args, plugin: Plugin, config: dict, source: dict):
    data = source['data']
    meta = dict(method=plugin.name, method_version=plugin.version,
                code_sha256=plugin.code_sha256,
                optimizer_api_sha256=sha256(Path(__file__).with_name('api.py')),
                runner_sha256=sha256(Path(__file__)), model=args.model,
                condition=args.condition, repeat=args.repeat, representation=args.representation,
                rotation_dim=data.k, configuration=config,
                seed=args.seed, fit_max_subjects=args.fit_max_subjects,
                whiten_floor=args.whiten_floor,
                source_latent_sha256=source['latent_sha'],
                source_checkpoint_sha256=source['checkpoint_sha'],
                no_evaluate=bool(args.no_evaluate),
                source_dataset_sha256=source['dataset_sha'],
                fit_ids_sha256=hashlib.sha256(source['chosen'].tobytes()).hexdigest())
    fingerprint = _digest_json(meta)
    base = source['root']/'rotations'/plugin.name/args.condition/f'repeat{args.repeat:02d}'
    folder = base/f'run_{fingerprint}'
    if (folder/'rotation__complete.json').exists() and not args.overwrite:
        print(f'[SKIP] {plugin.name}: exact experiment already exists at {folder}', flush=True)
        return 'skipped', folder
    if args.dry_run:
        print(f'[PLAN] {plugin.name} [{fingerprint}] config={json.dumps(config,sort_keys=True)}', flush=True)
        return 'planned', folder
    # Do not clobber a successfully finished run by default, even if other methods fail.
    folder.mkdir(parents=True, exist_ok=True)
    result = _invoke(plugin, data, config)
    R = result.R
    A = R @ source['W']
    offset = -A @ source['pca'].mean_
    U = source['z'] @ A.T + offset
    # Check exactly the FIT rows (not all participants) against the plugin convention.
    if not np.allclose(U[source['chosen']], data.rotated(R), atol=1e-6):
        raise RuntimeError('Z->U affine rotation does not match q@R.T')
    np.savez_compressed(folder/'rotation__matrix.npz', R=R, A=A, offset=offset,
                        pca_components=source['pca'].components_,
                        pca_eigenvalues=source['pca'].explained_variance_,
                        pca_mean=source['pca'].mean_, fit_ids=source['chosen'],
                        source_z_sha256=np.array(source['latent_sha']),
                        source_dataset_sha256=np.array(source['dataset_sha']),
                        method=np.array(plugin.name),
                        method_version=np.array(plugin.version))
    np.savez_compressed(folder/'rotation__U.npz', U=U, train_ids=source['tr'],
                        validation_ids=source['val'], test_ids=source['test'])
    if result.extra_arrays:
        np.savez_compressed(folder/'rotation__method_arrays.npz',
                            **{k:np.asarray(v) for k,v in result.extra_arrays.items()})
    full = {**meta, 'fingerprint':fingerprint, 'method_description':plugin.description,
            'fit_subject_count':int(data.n_subjects), 'plugin_source':str(plugin.path),
            'optimizer_result':_jsonable(result.info), 'state':'fit_saved'}
    (folder/'rotation__fit.json').write_text(json.dumps(full, indent=2, allow_nan=False),encoding='utf-8')
    # Theta is only accessed after successful persistence. No ground-truth parameter
    # vector is passed to any plugin or the optimization engine.
    if not args.no_evaluate:
        from .analysis import param_names
        labels = param_names(args.model)
        report = evaluate_after_freeze(source['dataset'].parameters, U, source['test'],
                                       folder, plugin.name, labels)
        print(f'[OK] {args.model} {plugin.name}: TEST matched r²={report["matched_mean_r2"]:.4f}; {folder}', flush=True)
    else:
        print(f'[OK] {args.model} {plugin.name}: fit saved (no theta evaluation); {folder}', flush=True)
    (folder/'rotation__complete.json').write_text(
        json.dumps({'fingerprint':fingerprint,'evaluated':not args.no_evaluate},indent=2),
        encoding='utf-8')
    return 'completed', folder


def summarize(root: Path):
    dirs = sorted(root.glob('*/**/run_*/rotation__test_evaluation.json'))
    rows = []
    for report_path in dirs:
        try:
            metrics = json.loads(report_path.read_text(encoding='utf-8'))
            fit = json.loads((report_path.parent/'rotation__fit.json').read_text(encoding='utf-8'))
            rows.append(dict(model=fit['model'], method=fit['method'],
                             method_version=fit['method_version'],
                             condition=fit['condition'], repeat=fit['repeat'],
                             rotation_dim=fit['rotation_dim'],
                             fingerprint=fit['fingerprint'],
                             matched_mean_r2=metrics['matched_mean_r2'],
                             matched_min_r2=metrics['matched_min_r2'],
                             max_offtarget_abs_r=metrics['max_offtarget_abs_r'],
                             mean_offtarget_r2=metrics['mean_offtarget_r2'],
                             folder=str(report_path.parent)))
        except (KeyError, ValueError, OSError) as exc:
            print(f'[WARN] Cannot summarize {report_path}: {exc}')
    report_file = root/'rotation__summary.csv'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with report_file.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f'[SUMMARY] {len(rows)} completed evaluations -> {report_file}')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description='Extensible independent rotation-method platform')
    p.add_argument('--model', choices=['C1','C2','C3','CR3'])
    p.add_argument('--condition', help='Frozen S4 condition tag')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--representation', choices=['pca','full'], default='pca')
    p.add_argument('--rotation-dim', type=int)
    p.add_argument('--method', action='append', default=[], help='Repeat --method for several modules')
    p.add_argument('--methods', nargs='+', default=[], help='Method names, or all')
    p.add_argument('--plugin-dir', action='append', default=[], help='Additional local directory with standalone plugins')
    p.add_argument('--list-methods', action='store_true')
    p.add_argument('--describe-method')
    p.add_argument('--plan', type=Path, help='JSON array of {method,config} experiments')
    p.add_argument('--config-file', type=Path, help='JSON with defaults and per-method settings')
    p.add_argument('--set', action='append', default=[], metavar='[METHOD.]PARAM=JSON_VALUE')
    p.add_argument('--starts', type=int, default=None)
    p.add_argument('--maxiter', type=int, default=None)
    p.add_argument('--seed', type=int, default=31)
    p.add_argument('--fit-max-subjects', type=int, default=2500)
    p.add_argument('--whiten-floor', type=float, default=1e-5)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-evaluate', action='store_true')
    p.add_argument('--summary', action='store_true', help='Rebuild CSV from saved TEST results')
    args = p.parse_args(argv)
    paths = discover_paths([BUILTINS] + [Path(s).resolve() for s in args.plugin_dir])
    if args.list_methods:
        for name, path in sorted(paths.items()):
            info = inspect_metadata(path)
            print(f'{name:20s} v{str(info.get("VERSION", "?")):<7s} '
                  f'{info.get("DESCRIPTION", "(dynamic)")}  [{path}]')
        return
    if args.describe_method:
        path = paths.get(args.describe_method)
        if path is None:
            p.error(f'Method not found: {args.describe_method}')
        plug = load_plugin(path)
        print(json.dumps(dict(name=plug.name, version=plug.version,
                              description=plug.description, defaults=plug.defaults,
                              path=str(plug.path), code_sha256=plug.code_sha256), indent=2))
        return
    if not args.model:
        p.error('--model is required for a run or summary')
    if args.summary:
        summarize(ROOT/'outputs'/'choice'/args.model/'rotations')
        return
    if not args.condition:
        p.error('--condition is required for a run')
    methods = args.method + args.methods
    if 'all' in methods:
        if len(methods) != 1:
            p.error('"all" cannot be mixed with individual methods')
        methods = sorted(paths)
    plan = _read_plan(args.plan, methods)
    specs, load_failed = [], []
    for item in plan:
        method = item['method']
        if method not in paths:
            raise KeyError(f'Unknown method {method!r}; available: {", ".join(sorted(paths))}')
        try:
            plugin = load_plugin(paths[method])
            config = _configs(plugin, args, item)
            specs.append((plugin, config))
        except Exception as exc:
            if not args.continue_on_error:
                raise
            load_failed.append((method, str(exc)))
            print(f'[IMPORT ERROR] {method}: {exc}', file=sys.stderr, flush=True)
    # Load aligned ZXY exactly once even for hundreds of methods.
    source = None if args.dry_run else _build_source(args)
    completed, skipped, failed = 0, 0, load_failed
    for plugin, config in specs:
        try:
            if args.dry_run:
                print(f'[PLAN] {plugin.name} v{plugin.version}: {json.dumps(config,sort_keys=True)}')
                continue
            status, _ = _run_one(args, plugin, config, source)
            completed += status == 'completed'
            skipped += status == 'skipped'
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed.append((plugin.name, str(exc)))
            print(f'[ERROR] {plugin.name}: {exc}\n{traceback.format_exc()}', file=sys.stderr, flush=True)
    if not args.dry_run:
        summarize(ROOT/'outputs'/'choice'/args.model/'rotations')
        print(f'[DONE] completed={completed} skipped={skipped} failed={len(failed)}')
    if failed:
        raise SystemExit(f'{len(failed)} plugin experiment(s) failed: {failed}')

if __name__ == '__main__':
    main()