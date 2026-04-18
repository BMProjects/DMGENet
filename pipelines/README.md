# Pipelines

This directory contains the end-to-end experiment entry scripts.

## Main entrypoint

- `run_unified_pipeline.sh`
  End-to-end orchestration for dataset split preparation, graph construction,
  DMGENet base-model training, RL input preparation, and graph-level integration.

At the moment this repository exposes a single public orchestration script:

- `run_unified_pipeline.sh`

Start there to reproduce the full DMGENet workflow.
