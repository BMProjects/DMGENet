# Pipelines

This directory contains the public-facing experiment entrypoints.

The scripts here are intended to orchestrate the paper-aligned workflow rather
than to preserve historical experiment branches.

## Main entrypoint

- `run_unified_pipeline.sh`
  End-to-end orchestration for the current unified protocol:
  local split preparation from compact station panels, graph construction, DMGENet base-model
  training, RL input preparation, and graph-level integration.

At the moment this repository exposes a single public orchestration script:

- `run_unified_pipeline.sh`

If you are automating reproduction or building a clean benchmark run, start there.
