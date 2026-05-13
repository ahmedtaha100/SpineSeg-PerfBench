#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_controller import load_state, status_table  # noqa: E402


def main() -> int:
    state = load_state()
    print(status_table(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
