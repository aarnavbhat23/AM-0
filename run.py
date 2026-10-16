"""Run the preregistered AM-0 diagnostic; preserve data and all predictions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import tracemalloc

import numpy as np

from assembly_memory.data import dataset, data_hash
from assembly_memory.models import Assembly, SignedAssembly, make_model

ROOT = Path(__file__).resolve().parent


def source_info():
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    # Previous committed result directories are fine; untracked outputs do not change source.
    dirty = [line for line in status.splitlines() if not line[3:].startswith('results/')]
    if dirty:
        raise RuntimeError('Commit source/protocol before running: ' + repr(dirty))
    hashes = {}
    for pattern in ('assembly_memory/*.py', 'tests/*.py', 'configs/*.json', 'docs/*.md', 'run.py'):
        for p in ROOT.glob(pattern):
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': hashes, 'python': platform.python_version(), 'numpy': np.__version__,
        'platform': platform.platform(), 'machine': platform.machine(),
        'threads': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
    }


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def prepare(cfg, family, seed, out):
    start = time.perf_counter()
    sets = {split: dataset(family, seed, split, cfg[count], cfg['length'], cfg['warmup'])
            for split, count in [('train', 'train_sequences'), ('validation', 'validation_sequences'),
                                 ('test', 'test_sequences')]}
    elapsed = time.perf_counter() - start
    np.savez_compressed(out / f'data_{family}_{seed}.npz',
                        **{f'{split}_{key}': value for split, data in sets.items() for key, value in data.items()})
    return sets, {'hashes': {s: data_hash(d) for s, d in sets.items()}, 'preprocess_seconds': elapsed}


def fit(name, eta, train, mode, cap, seed):
    start = time.perf_counter()
    model = make_model(name, eta)
    x, true_y = train['x'], train['y']
    targets = true_y
    if name == 'assembly_shuffled':
        targets = np.random.default_rng(seed + 20000).permutation(true_y)
    steps = 0
    prequential = []
    while name != 'assembly_untrained':
        if mode == 'one_pass' and steps >= len(x):
            break
        if mode == 'time_cap' and time.perf_counter() - start >= cap:
            break
        j = steps % len(x)
        prediction = model.predict(x[j])
        if steps < len(x):
            prequential.append((prediction - true_y[j]) ** 2)
        model.update(x[j], targets[j])
        steps += 1
    elapsed = time.perf_counter() - start
    stats = {'training_seconds': elapsed, 'updates': steps,
             'unique_examples': min(steps, len(x)), 'all_data_seen': steps >= len(x),
             'overrun_seconds': max(0., elapsed - cap) if mode == 'time_cap' else 0.,
             'state_bytes': model.state_bytes(), 'active_features': int(model.active.sum()),
             'births': model.births, 'evictions': model.evictions}
    if name == 'assembly_untrained':
        stats['all_data_seen'] = None
    if isinstance(model, Assembly):
        stats['invariants'] = model.invariants()
    return model, stats, np.array(prequential)


def evaluate(model, data, scale):
    start = time.perf_counter()
    pred = np.array([model.predict(x) for x in data['x']])
    seconds = time.perf_counter() - start
    squared = (pred - data['y']) ** 2
    ids, held = data['sequence'], data['withheld']
    return pred, {
        'nmse': float(squared.mean() / scale),
        'withheld_nmse': float(squared[held].mean() / scale) if held.any() else None,
        'withheld_count': int(held.sum()),
        'sequence_nmse': [float(squared[ids == i].mean() / scale) for i in np.unique(ids)],
        'inference_seconds': seconds,
    }


def audit():
    rng = np.random.default_rng(818)
    a, b = Assembly(eta=0.1, capacity=10), SignedAssembly(eta=0.1, capacity=10)
    errors = []
    max_mass, min_amount = 0., 0.
    for i in range(2000):
        x, y = np.r_[1., rng.uniform(-1, 1, 7)], rng.normal()
        errors.append(abs(a.predict(x) - b.predict(x)))
        a.update(x, y)
        b.update(x, y)
        if not np.array_equal(a.active, b.active):
            raise AssertionError('support mismatch in equivalence audit')
        inv = a.invariants()
        max_mass = max(max_mass, inv['max_mass_error'])
        min_amount = min(min_amount, inv['min_amount'])
    tracemalloc.start()
    model = Assembly()
    for i in range(100):
        model.update(np.r_[1., rng.uniform(-1, 1, 7)], rng.normal())
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'equivalence_steps': 2000, 'max_prediction_difference': max(errors),
            'max_coefficient_difference': float(np.max(np.abs(a.plus - a.minus - b.w))),
            'max_occupancy_difference': float(np.max(np.abs(a.plus + a.minus - b.m))),
            'max_mass_error': max_mass, 'min_amount': min_amount,
            'births': a.births, 'evictions': a.evictions,
            'assembly_python_traced_current_bytes': current,
            'assembly_python_traced_peak_bytes': peak,
            'workspace_note': 'Representative Python/NumPy-traced allocations, not process peak RSS; excludes dataset/interpreter.'}


def development(cfg, out):
    records, data_records, predictions = [], {}, {}
    trained = cfg['models'][:-2]
    for family in cfg['development_families']:
        for seed in cfg['development_seeds']:
            sets, meta = prepare(cfg, family, seed, out)
            data_records[f'{family}_{seed}'] = meta
            scale = max(float(np.var(sets['train']['y'])), 0.01)
            for name in trained:
                for eta in cfg['learning_rates']:
                    model, stats, pre = fit(name, eta, sets['train'], 'one_pass', 0, seed)
                    pred, metrics = evaluate(model, sets['validation'], scale)
                    key = f'{family}_{seed}_{name}_{eta}'
                    predictions[key] = pred
                    records.append(dict(family=family, seed=seed, model=name, eta=eta,
                                        scale=scale, **stats, **metrics))
            print(f'development: {family} seed {seed}', flush=True)
    selected = {}
    for name in trained:
        selected[name] = min(cfg['learning_rates'], key=lambda eta:
                            np.mean([r['nmse'] for r in records if r['model'] == name and r['eta'] == eta]))
    selected['assembly_shuffled'] = selected['assembly']
    selected['assembly_untrained'] = selected['assembly']
    np.savez_compressed(out / 'validation_predictions.npz', **predictions)
    dump(out / 'development.json', {'records': records, 'data': data_records, 'selected_rates': selected})
    dump(out / 'selected_rates.json', selected)
    return selected


def evaluation(cfg, out, selection_path):
    rates = json.loads(selection_path.read_text())
    dump(out / 'selected_rates.json', rates)
    records, data_records, predictions, curves = [], {}, {}, {}
    for family in cfg['families']:
        for seed in cfg['evaluation_seeds']:
            sets, meta = prepare(cfg, family, seed, out)
            data_records[f'{family}_{seed}'] = meta
            scale = max(float(np.var(sets['train']['y'])), 0.01)
            for mode in ('one_pass', 'time_cap'):
                for name in cfg['models']:
                    model, stats, pre = fit(name, rates[name], sets['train'], mode, cfg['time_cap_seconds'], seed)
                    if stats['state_bytes'] > cfg['state_cap_bytes']:
                        raise AssertionError(f'state budget exceeded: {name}')
                    pred, metrics = evaluate(model, sets['test'], scale)
                    key = f'{family}_{seed}_{mode}_{name}'
                    predictions[key] = pred
                    curves[key] = pre
                    records.append(dict(family=family, seed=seed, mode=mode, model=name, eta=rates[name],
                                        scale=scale, **stats, **metrics,
                                        prequential_quarters=[float(np.mean(q) / scale) for q in np.array_split(pre, 4)] if len(pre) else []))
            print(f'evaluation: {family} seed {seed}', flush=True)
            # Preserve progress if interrupted; never silently resume or append a new run.
            dump(out / 'measurements.json', {'records': records, 'data': data_records,
                 'selection_file_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest()})
            np.savez_compressed(out / 'predictions.npz', **predictions)
            np.savez_compressed(out / 'prequential.npz', **curves)
    dump(out / 'audit.json', audit())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('development', 'evaluation'), required=True)
    parser.add_argument('--selected', type=Path)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'configs/diagnostic.json').read_text())
    source = source_info()
    if args.phase == 'evaluation' and args.selected is None:
        parser.error('--selected is required for evaluation')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'results' / f'{args.phase}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'metadata.json', dict(source, config=cfg, phase=args.phase,
         started_at_utc=datetime.now(timezone.utc).isoformat(),
         user_timezone='America/Los_Angeles', status='running'))
    print(out, flush=True)
    if args.phase == 'development':
        print(json.dumps(development(cfg, out), sort_keys=True), flush=True)
    else:
        evaluation(cfg, out, args.selected.resolve())
    meta = json.loads((out / 'metadata.json').read_text())
    meta.update(status='complete', finished_at_utc=datetime.now(timezone.utc).isoformat())
    dump(out / 'metadata.json', meta)


if __name__ == '__main__':
    main()
