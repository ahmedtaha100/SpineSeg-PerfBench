.PHONY: help install smoke test prepare train benchmark robustness optimize tables plots verify freeze all clean

CKPT ?= outputs/runs/segresnet_baseline/checkpoint.pt
CONFIG ?= opt_baseline
MODEL ?= segresnet

help:
	@echo "Targets:"
	@echo "  install     Install the package in editable mode"
	@echo "  smoke       Run the CPU synthetic smoke pipeline"
	@echo "  test        Run pytest"
	@echo "  verify      Verify artifacts/frozen"
	@echo "  tables      Regenerate result tables from outputs/"
	@echo "  plots       Regenerate result figures from outputs/"
	@echo "  all         Run scripts/run_all.sh"

install:
	python -m pip install -e .

smoke:
	python scripts/smoke_test.py --all --smoke

test:
	pytest tests/ -v

prepare:
	@if [ -z "$(DATA)" ]; then echo "ERROR: set DATA=/path/to/dataset for make prepare"; exit 1; fi
	python scripts/prepare_data.py --verse-root "$(DATA)"

train:
	python scripts/train.py model=$(MODEL)

benchmark:
	python scripts/benchmark.py --config $(CONFIG) --checkpoint "$(CKPT)"

robustness:
	python scripts/robustness.py --config $(CONFIG) --checkpoint "$(CKPT)"

optimize:
	python scripts/benchmark.py --config opt_data_pipeline --checkpoint "$(CKPT)" --sweep
	python scripts/benchmark.py --config opt_amp --checkpoint "$(CKPT)" --sweep
	python scripts/benchmark.py --config opt_compile --checkpoint "$(CKPT)"
	python scripts/benchmark.py --config opt_all --checkpoint "$(CKPT)"

tables:
	python scripts/make_tables.py

plots:
	python scripts/make_plots.py

verify:
	python scripts/verify_artifacts.py artifacts/frozen

freeze:
	python scripts/freeze_artifacts.py

all:
	bash scripts/run_all.sh

clean:
	rm -rf outputs artifacts/smoke_frozen artifacts/local_frozen artifacts/demo .pytest_cache .mypy_cache .ruff_cache
