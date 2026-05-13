from __future__ import annotations


def _clean_cell(value: str) -> str:
    return value.strip().strip("`")


def _is_header_or_separator(run_id: str) -> bool:
    return run_id.lower() == "run_id" or set(run_id).issubset({"-", ":"})


def ledger_entries(ledger: str) -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    for line in ledger.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cols = [col.strip() for col in stripped.strip("|").split("|")]
        if len(cols) < 4:
            continue
        run_id = _clean_cell(cols[0])
        if _is_header_or_separator(run_id):
            continue
        json_path = _clean_cell(cols[3])
        if run_id and json_path.endswith(".json"):
            entries.add((run_id, json_path))
    return entries


def ledger_json_paths(ledger: str) -> list[str]:
    return [json_path for _, json_path in sorted(ledger_entries(ledger))]


def ledger_run_ids(ledger: str) -> set[str]:
    return {run_id for run_id, _ in ledger_entries(ledger)}
