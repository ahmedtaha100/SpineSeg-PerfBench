#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from spineseg_perfbench.data.manifests import discover_pairs, synthetic_manifest, write_manifest_and_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SpineSeg-PerfBench manifests and deterministic splits.")
    p.add_argument("--synthetic", action="store_true", help="Generate tiny synthetic NIfTI fixtures.")
    p.add_argument("--verse-root", default="", help="Root containing VerSe data.")
    p.add_argument("--ctspine1k-root", default="", help="Root containing CTSpine1K data.")
    p.add_argument("--image-glob", default=None, help="Override recursive image glob.")
    p.add_argument("--label-glob", default=None, help="Override recursive label glob.")
    p.add_argument("--output-dir", default="outputs/manifests")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.synthetic or args.smoke:
        shape = (16, 16, 16) if args.smoke else (32, 32, 32)
        df = synthetic_manifest("tests/fixtures/synthetic", n_cases=4, shape=shape)
    else:
        roots = []
        if args.verse_root:
            roots.append((Path(args.verse_root), "verse"))
        if args.ctspine1k_root:
            roots.append((Path(args.ctspine1k_root), "ctspine1k"))
        if not roots:
            raise SystemExit("ERROR: provide --synthetic, --verse-root, or --ctspine1k-root")
        df = discover_pairs(roots, image_glob=args.image_glob, label_glob=args.label_glob)
    paths = write_manifest_and_splits(df, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
