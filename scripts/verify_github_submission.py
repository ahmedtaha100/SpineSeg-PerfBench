#!/usr/bin/env python3
"""Verify the GitHub-clean SpineSeg-PerfBench artifact subset.

This verifier is intentionally separate from the full frozen-bundle provenance:
`verify_artifacts.py` verifies a checksum-described bundle, while this script also
checks GitHub-specific omissions and privacy/size boundaries.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, subprocess, sys
from pathlib import Path

RAW_SUFFIXES = ('.nii', '.nii.gz', '.mha', '.mhd', '.nrrd', '.dcm', '.dicom')
CHECKPOINT_SUFFIXES = ('.pt', '.pth', '.ckpt')
OS_METADATA_NAMES = {'.DS_Store', 'Thumbs.db'}
MAX_GITHUB_BYTES = 100 * 1024 * 1024
REQUIRED = [
    'SPEC.md', 'RUNS.md', 'ARTIFACT_INDEX.json', 'checksums.sha256',
    'outputs/benchmarks', 'outputs/manifests', 'outputs/profiles',
    'outputs/tables', 'outputs/figures',
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('bundle', nargs='?', default='artifacts/frozen')
    args = ap.parse_args()
    root = Path(args.bundle)
    if not root.exists():
        raise SystemExit(f'Missing bundle: {root}')
    missing = [p for p in REQUIRED if not (root / p).exists()]
    if missing:
        raise SystemExit(f'Missing required paths: {missing}')
    raw = [p for p in root.rglob('*') if p.is_file() and any(str(p).lower().endswith(s) for s in RAW_SUFFIXES)]
    if raw:
        raise SystemExit(f'Raw medical-image-like files present: {raw[:10]}')
    checkpoints = [
        p for p in root.rglob('*') if p.is_file() and any(str(p).lower().endswith(s) for s in CHECKPOINT_SUFFIXES)
    ]
    if checkpoints:
        raise SystemExit(f'Model checkpoint files should not be in the GitHub-clean bundle: {checkpoints[:10]}')
    os_metadata = [p for p in root.rglob('*') if p.name in OS_METADATA_NAMES or '__MACOSX' in p.parts]
    if os_metadata:
        raise SystemExit(f'OS metadata files should not be in the GitHub-clean bundle: {os_metadata[:10]}')
    huge = [p for p in root.rglob('*') if p.is_file() and p.stat().st_size > MAX_GITHUB_BYTES]
    if huge:
        raise SystemExit(f'Files over 100 MB present: {huge}')
    trace_files = list((root / 'outputs' / 'profiles').glob('**/trace.json'))
    omission_note = root.parent / 'PROFILE_TRACE_OMITTED.md'
    if trace_files:
        for p in trace_files:
            if p.stat().st_size > MAX_GITHUB_BYTES:
                raise SystemExit(f'Oversized trace present: {p}')
    elif not omission_note.exists():
        raise SystemExit('Profiler trace omitted but artifacts/PROFILE_TRACE_OMITTED.md is missing')
    # checksum validation
    ck = root / 'checksums.sha256'
    for line in ck.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        candidates = [root / rel, Path(rel)]
        path = None
        escaped_existing_path = False
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                candidate.resolve().relative_to(root.resolve())
            except ValueError:
                escaped_existing_path = True
                continue
            path = candidate
            break
        if path is None and escaped_existing_path:
            raise SystemExit(f'Checksum entry escapes bundle root: {rel}')
        if path is None:
            raise SystemExit(f'Checksum entry missing file: {rel}')
        actual = sha256_file(path)
        if actual != digest:
            raise SystemExit(f'Checksum mismatch: {rel}')
    # benchmark rows
    rows = sorted((root / 'outputs' / 'benchmarks').glob('*.json'))
    if len(rows) != 48:
        raise SystemExit(f'Expected 48 benchmark JSON rows, found {len(rows)}')
    try:
        from spineseg_perfbench.utils.schema import validate_run_row
        for p in rows:
            validate_run_row(json.loads(p.read_text()))
    except Exception as exc:
        raise SystemExit(f'Benchmark schema validation failed: {exc}')
    rob = sorted((root / 'outputs').glob('robustness_results*.csv'))
    if not rob:
        raise SystemExit('No robustness CSVs found')
    profile_summaries = list((root / 'outputs' / 'profiles').glob('**/operator_summary.csv')) + list((root / 'outputs' / 'profiles').glob('**/phase_summary.json'))
    if len(profile_summaries) < 2:
        raise SystemExit('Profile summaries missing')
    for run in ['segresnet_baseline', 'unet_baseline']:
        if not (root / 'outputs' / 'runs' / run / 'metrics.json').exists():
            raise SystemExit(f'Missing training metrics for {run}')
    print('GitHub-clean artifact subset verification PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
