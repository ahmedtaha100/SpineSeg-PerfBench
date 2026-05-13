# Profiler Trace Omitted from GitHub-Clean Bundle

The full local frozen artifact bundle contained a PyTorch profiler `trace.json` file. It is intentionally omitted from this GitHub/submission-clean folder because it exceeds GitHub's 100 MB per-file limit.

Included instead:
- `artifacts/frozen/outputs/profiles/*/operator_summary.csv`
- `artifacts/frozen/outputs/profiles/*/phase_summary.json`

The active `artifacts/frozen/checksums.sha256` and `ARTIFACT_INDEX.json` describe the GitHub-clean subset in this folder. The trace can be regenerated on GPU with:

```bash
python scripts/profile.py --config opt_baseline --checkpoint outputs/runs/segresnet_baseline/checkpoint.pt
```
