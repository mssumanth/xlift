# xLift Makefile — follows spec §12 execution plan
# Usage: make <target> [COHORTS="C2_frontier,C1_easy"] [GPUS="0,1"] [NGPUS=8]

PYTHON      := python
CLI         := $(PYTHON) -m xlift.cli
COHORTS     ?= C2_frontier,C1_easy
GPUS        ?= 0,1
NGPUS       ?= 8
N_FOUND     ?= 4000
NCAND       ?= 1000

.PHONY: help setup smoke data foundation eval-base synth cohorts metrics \
        train gradnorm eval analyze sweep status test clean

help:
	@echo "xLift targets (run in order for first-time setup):"
	@echo "  make setup        — check env, model, API keys"
	@echo "  make smoke        — GATE 1: 10-step TRL run on 16 tasks"
	@echo "  make data         — load GSM8K train+test jsonl"
	@echo "  make foundation   — vLLM k=16 rollouts → foundation.jsonl"
	@echo "  make eval-base    — GATE 2: base model accuracy (~0.45)"
	@echo "  make synth        — Claude synthetic tasks for C7"
	@echo "  make cohorts      — build C1-C7 from foundation index"
	@echo "  make metrics      — compute all cheap signals"
	@echo "  make train        — train COHORTS=... on GPUS=..."
	@echo "  make gradnorm     — mid-cost oracle for all cohorts"
	@echo "  make eval         — evaluate checkpoints, compute lift"
	@echo "  make analyze      — table + validation + all 4 plots"
	@echo "  make sweep        — parallel sweep remaining cohorts (NGPUS=8)"
	@echo "  make status       — print completion status"
	@echo "  make test         — run unit tests (no model needed)"
	@echo ""
	@echo "Hackathon fast path:"
	@echo "  make setup && make smoke && make data && make foundation && make eval-base"
	@echo "  make cohorts && make metrics"
	@echo "  make train COHORTS=C2_frontier,C1_easy GPUS=0,1"
	@echo "  make eval COHORTS=C2_frontier,C1_easy && make analyze"

setup:
	$(CLI) setup

smoke:
	$(CLI) smoke --steps 10 --n-tasks 16

data:
	$(CLI) data

foundation:
	$(CLI) foundation --n $(N_FOUND)

eval-base:
	$(CLI) eval-base

synth:
	$(CLI) synth --n-cand $(NCAND)

cohorts:
	$(CLI) cohorts

metrics:
	$(CLI) metrics --cohort-names all

metrics-force:
	$(CLI) metrics --cohort-names all --force

metrics-one:
	$(CLI) metrics --cohort-names "$(COHORTS)"

train:
	$(CLI) train --cohort-names "$(COHORTS)" --gpus "$(GPUS)"

gradnorm:
	$(CLI) gradnorm --cohort-names all

eval:
	@if [ "$(COHORTS)" = "all" ]; then \
		$(CLI) evaluate --cohort-names all; \
	else \
		$(CLI) evaluate --cohort-names "$(COHORTS)"; \
	fi

analyze:
	$(CLI) analyze

sweep:
	$(CLI) sweep --n-gpus $(NGPUS)

status:
	$(CLI) status

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	@echo "Remove artifacts? This deletes all rollouts, metrics, and training outputs."
	@echo "Run: rm -rf artifacts/ results/"
