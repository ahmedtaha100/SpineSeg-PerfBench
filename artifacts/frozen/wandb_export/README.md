# WandB Export Snapshot

This directory is a static export of the WandB project
`ahmedtaeha2/spineseg-perfbench`. It exists because the source project is hosted under a
WandB team entity whose current plan does not expose a working public
visibility toggle. Reviewers can inspect this export without WandB
authentication.

The frozen benchmark JSON/CSV rows elsewhere in `artifacts/frozen/` remain the
canonical source for reported values. This directory preserves the WandB
visualization layer as exported run configs, summaries, scalar histories, and
available system metrics.

## Contents

- `index.json` lists the 49 exported runs and points to each run
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

The export is a snapshot, not an interactive dashboard. `wandb_report.pdf` is a
generated static PDF summary built from the CSV/JSON files in this directory.
If a code editor opens the PDF as raw `%PDF-1.4` bytes instead of rendering it,
open `wandb_report_preview.md`; it embeds PNG renders of the same eight pages.
