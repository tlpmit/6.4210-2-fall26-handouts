"""Supplied bookkeeping for repeatable notebook runs; no experiments on import."""
import hashlib
import json
from pathlib import Path
import tempfile


def digest(path):
    """SHA256 of a file's contents, independent of its modification time."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cached(folder, *, name, settings, inputs, outputs, run, retrain=False):
    """Reuse a completed JSON result only when inputs and saved artifacts match.

    `run()` performs the work and returns a JSON-serializable result. `inputs`
    lists code/data files; `outputs` lists owned filenames or glob patterns
    relative to `folder`. Different runs sharing a folder must own disjoint
    outputs. A missing/stale/interrupted result requires explicit `retrain=True`,
    which archives this run's earlier files before starting again.
    """
    folder = Path(folder)
    receipt = folder / f'.{name}.cache.json'
    specification = dict(settings=settings, inputs={str(Path(p).resolve()): digest(p)
        for p in sorted(set(map(Path, inputs)))})
    # Normalize tuples, etc. to the same representation read back from JSON.
    specification = json.loads(json.dumps(specification, sort_keys=True, allow_nan=False))
    owned = sorted({p for pattern in outputs for p in folder.glob(pattern) if p.is_file()})
    previous = None
    if receipt.exists():
        try:
            previous = json.loads(receipt.read_text())
        except (ValueError, OSError):
            pass
    if not retrain and previous and previous.get('status') == 'complete':
        artifacts = previous['artifacts']
        intact = all((folder / p).is_file() and digest(folder / p) == checksum
                     for p, checksum in artifacts.items())
        if previous['specification'] == specification and intact:
            print(f'Using saved {name}: {folder}', flush=True)
            return previous['result']
    if receipt.exists() or owned:
        if not retrain:
            raise RuntimeError(
                f'{name} has changed, incomplete, or unverified output in {folder}. '
                'Use retrain=True to archive it and run again, or choose another output folder.')
        archive_root = folder.parent / '_previous_runs'
        archive_root.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkdtemp(prefix=f'{folder.name}-{name}-', dir=archive_root))
        for path in [*owned, *([receipt] if receipt.exists() else [])]:
            destination = archive / path.relative_to(folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
        print(f'Archived previous {name}: {archive}', flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(dict(status='running', specification=specification), indent=2) + '\n')
    print(f'Running {name}: {folder}', flush=True)
    result = run()
    artifacts = {str(p.relative_to(folder)): digest(p)
                 for pattern in outputs for p in sorted(folder.glob(pattern)) if p.is_file()}
    payload = dict(status='complete', specification=specification, artifacts=artifacts, result=result)
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(receipt)
    return result
