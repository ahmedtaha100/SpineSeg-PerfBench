#!/usr/bin/env python3
"""Export a WandB project into the frozen artifact bundle.

The export is a static snapshot of run configs, summaries, scalar histories,
and available system metrics. It does not run benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--bundle", default="artifacts/frozen")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    return parser.parse_args()


def require_env(args: argparse.Namespace) -> None:
    missing = []
    if not os.environ.get("WANDB_API_KEY"):
        missing.append("WANDB_API_KEY")
    if not args.entity:
        missing.append("WANDB_ENTITY or --entity")
    if not args.project:
        missing.append("WANDB_PROJECT or --project")
    if missing:
        raise SystemExit(f"Missing required WandB inputs: {', '.join(missing)}")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    try:
        isoformat = getattr(value, "isoformat")
    except Exception:
        isoformat = None
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def flatten_summary(summary: Any) -> dict[str, Any]:
    data = dict(summary)
    return {
        key: value
        for key, value in data.items()
        if not key.startswith("_") and key not in {"_wandb"}
    }


def retry(label: str, retries: int, fn: Callable[[], Any]) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # WandB public API raises several transient types.
            last_exc = exc
            if attempt == retries:
                break
            wait = min(2 ** attempt, 10)
            print(f"{label} failed on attempt {attempt}/{retries}: {exc}; retrying in {wait}s")
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    else:
        keys = ["_step"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: jsonable(value) for key, value in row.items()})


def dataframe_to_csv(path: Path, frame: Any) -> int:
    if frame is None or getattr(frame, "empty", True):
        write_rows_csv(path, [])
        return 0
    frame.to_csv(path, index=False)
    return len(frame)


def scan_history_rows(run: Any, retries: int) -> list[dict[str, Any]]:
    def load() -> list[dict[str, Any]]:
        return [dict(row) for row in run.scan_history(page_size=1000)]

    try:
        return retry(f"scan_history {run.id}", retries, load)
    except Exception as exc:
        print(f"scan_history failed for {run.id}: {exc}; falling back to sampled history")

    def load_sampled() -> list[dict[str, Any]]:
        frame = run.history(samples=100000)
        if frame is None or frame.empty:
            return []
        return frame.to_dict(orient="records")

    return retry(f"history {run.id}", retries, load_sampled)


def system_history_frame(run: Any, retries: int) -> Any:
    def load() -> Any:
        return run.history(stream="system", samples=100000)

    try:
        return retry(f"system_history {run.id}", retries, load)
    except Exception as exc:
        print(f"system history unavailable for {run.id}: {exc}")
        return None


def metadata_for_run(run: Any, entity: str, project: str) -> dict[str, Any]:
    url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{run.id}"
    return {
        "id": getattr(run, "id", None),
        "name": getattr(run, "name", None),
        "path": list(getattr(run, "path", []) or []),
        "url": url,
        "state": getattr(run, "state", None),
        "tags": list(getattr(run, "tags", []) or []),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "created_at": getattr(run, "created_at", None),
        "updated_at": getattr(run, "updated_at", None),
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "duration": getattr(run, "duration", None),
        "user": getattr(getattr(run, "user", None), "username", None),
        "notes": getattr(run, "notes", None),
    }


def classify_run(tags: list[str], name: str, config: dict[str, Any]) -> str:
    tags_lower = {str(tag).lower() for tag in tags}
    name_lower = name.lower()
    perturbation = str(config.get("perturbation", "")).lower()
    if name_lower in {"project_assets", "assets"}:
        return "other"
    if "training" in tags_lower or "train" in tags_lower or "train" in name_lower:
        return "training"
    if "robustness" in tags_lower or name_lower.startswith("robust_"):
        return "robustness"
    if perturbation and perturbation not in {"none", "clean"}:
        return "robustness"
    if any(tag.startswith("severity_") and tag != "severity_0" for tag in tags_lower):
        return "robustness"
    return "clean_inference"


def final_table_exclusion_reason(category: str, config: dict[str, Any]) -> str | None:
    metadata = config.get("optimization_metadata") or {}
    amp_dtype = str(metadata.get("amp_dtype", "")).lower()
    if category == "clean_inference" and config.get("optimization") == "amp" and amp_dtype == "fp32":
        return (
            "AMP-labeled clean row has effective amp_dtype=fp32, so it is retained "
            "as raw audit data but excluded from the rendered final clean table."
        )
    return None


def write_export_readme(path: Path, entity: str, project: str, run_count: int) -> None:
    path.write_text(
        f"""# WandB Export Snapshot

This directory is a static export of the WandB project
`{entity}/{project}`. It exists because the source project is hosted under a
WandB team entity whose current plan does not expose a working public
visibility toggle. Reviewers can inspect this export without WandB
authentication.

The frozen benchmark JSON/CSV rows elsewhere in `artifacts/frozen/` remain the
canonical source for reported values. This directory preserves the WandB
visualization layer as exported run configs, summaries, scalar histories, and
available system metrics.

## Contents

- `index.json` lists the {run_count} exported runs and points to each run
  directory.
- `runs/<run_id>/config.json` contains the run configuration.
- `runs/<run_id>/summary.json` contains final scalar summaries.
- `runs/<run_id>/history.csv` contains exported scalar history rows.
- `runs/<run_id>/system_metrics.csv` contains exported system metrics when
  available; it may contain only a header if WandB has no system stream for the
  run.
- `runs/<run_id>/metadata.json` contains run identity, tags, state, timestamps,
  and original WandB URL.

Clean inference exports: 11 raw -> 8 unique logical IDs after duplicate
collapse -> 7 rows in final rendered clean table after AMP-fp32 exclusion.

The export is a snapshot, not an interactive dashboard. A generated PDF summary
can be rebuilt from this export with
`python scripts/build_wandb_export_report.py`.
""",
        encoding="utf-8",
    )


def dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    args = parse_args()
    require_env(args)

    import wandb

    entity = str(args.entity)
    project = str(args.project)
    bundle = Path(args.bundle)
    export_root = bundle / "wandb_export"
    runs_root = export_root / "runs"

    if export_root.exists():
        shutil.rmtree(export_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=args.timeout)
    project_path = f"{entity}/{project}"
    runs = list(retry("list runs", args.retries, lambda: api.runs(project_path)))
    exported: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    category_counts = {"clean_inference": 0, "robustness": 0, "training": 0, "other": 0}

    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] exporting {run.id} {run.name}")
        run_dir = runs_root / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            config = dict(getattr(run, "config", {}) or {})
            summary = flatten_summary(getattr(run, "summary", {}) or {})
            metadata = metadata_for_run(run, entity, project)
            tags = list(metadata.get("tags", []) or [])
            category = classify_run(tags, str(metadata.get("name") or run.id), config)
            category_counts[category if category in category_counts else "other"] += 1
            exclusion_reason = final_table_exclusion_reason(category, config)

            history_rows = scan_history_rows(run, args.retries)
            system_rows = dataframe_to_csv(
                run_dir / "system_metrics.csv",
                system_history_frame(run, args.retries),
            )
            write_rows_csv(run_dir / "history.csv", history_rows)
            write_json(run_dir / "config.json", config)
            write_json(run_dir / "summary.json", summary)
            write_json(run_dir / "metadata.json", {**metadata, "category": category})

            item = {
                "run_id": run.id,
                "name": metadata.get("name"),
                "category": category,
                "tags": tags,
                "state": metadata.get("state"),
                "url": metadata.get("url"),
                "summary": summary,
                "history_rows": len(history_rows),
                "system_metric_rows": system_rows,
                "path": f"runs/{run.id}",
            }
            if exclusion_reason:
                item["excluded_from_final_table"] = True
                item["exclusion_reason"] = exclusion_reason
            exported.append(item)
        except Exception as exc:
            failed.append({"run_id": getattr(run, "id", "<unknown>"), "name": getattr(run, "name", ""), "error": str(exc)})
            print(f"FAILED {getattr(run, 'id', '<unknown>')}: {exc}")
        time.sleep(args.sleep_sec)

    export_index = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "entity": entity,
        "project": project,
        "project_url": f"https://wandb.ai/{entity}/{project}",
        "run_count": len(exported),
        "failed_count": len(failed),
        "category_counts": category_counts,
        "runs": exported,
        "failed": failed,
        "notes": "Static WandB API export. Frozen benchmark artifacts remain canonical.",
        "clean_inference_reconciliation": (
            "Clean inference exports: 11 raw -> 8 unique logical IDs after duplicate "
            "collapse -> 7 rows in final rendered clean table after AMP-fp32 exclusion."
        ),
    }
    write_json(export_root / "index.json", export_index)
    write_export_readme(export_root / "README.md", entity, project, len(exported))

    size_bytes = dir_size(export_root)
    print()
    print("WandB export summary")
    print(f"  project: {project_path}")
    print(f"  runs exported: {len(exported)}")
    print(f"  failures: {len(failed)}")
    print(f"  size bytes: {size_bytes}")
    print(f"  output: {export_root}")
    if failed:
        print("  failed runs:")
        for item in failed:
            print(f"    {item['run_id']}: {item['error']}")

    print()
    print("To build the static PDF summary from this export, run:")
    print("  python scripts/build_wandb_export_report.py")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
