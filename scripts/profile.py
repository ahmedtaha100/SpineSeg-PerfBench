#!/usr/bin/env python
from __future__ import annotations

if __name__ == "profile":
    # Compatibility shim: when this directory is on sys.path, stdlib cProfile
    # may import this file as `profile`. It only needs these attributes during
    # initialization. The real CLI is guarded below under __main__.
    def run(statement, filename=None, sort=-1):
        raise RuntimeError("scripts/profile.py is not the stdlib profile runner")

    def runctx(statement, global_vars=None, local_vars=None, filename=None, sort=-1):
        raise RuntimeError("scripts/profile.py is not the stdlib profile runner")

    class Profile:
        def run(self, cmd):
            return self

        def runctx(self, cmd, global_vars=None, local_vars=None):
            return self

        def runcall(self, func, /, *args, **kwargs):
            return func(*args, **kwargs)

        def print_stats(self, sort=-1):
            return self

        def dump_stats(self, file):
            return self

else:
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infer import run_inference  # noqa: E402

    from spineseg_perfbench.config import load_config
    from spineseg_perfbench.profiling.profiler import TorchProfiler
    from spineseg_perfbench.utils.hashing import stable_hash
    from spineseg_perfbench.utils.io import ensure_dir, write_json

    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Profile a small inference run.")
        p.add_argument("--config", default="opt_baseline")
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--smoke", action="store_true")
        return p.parse_args()

    def main() -> int:
        args = parse_args()
        cfg = load_config(args.config, smoke=args.smoke)
        run_id = f"profile_{stable_hash({'config': cfg, 'checkpoint': args.checkpoint})}"
        out_dir = ensure_dir(Path("outputs/profiles") / run_id)
        with TorchProfiler(out_dir) as prof:
            result = run_inference(args.checkpoint, cfg, smoke=args.smoke, profile_step=prof.step)
        write_json(out_dir / "phase_summary.json", result["phase_times_sec"])
        print(out_dir)
        return 0

    if __name__ == "__main__":
        raise SystemExit(main())
