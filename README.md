# ModuLab

Empirical benchmark: AM/FM narrowband decomposition vs STFT for speech-to-speech manipulation.

## Hypothesis

AM/FM trades ~10x analysis FLOPs for a compact, phase-explicit representation.
The win, if any, is conditional on downstream model scale and utterance length.

## Quick start

    modulab bench

Produces RESULTS.md and results/figures/narrowband_sweep.png.

## CLI

    modulab bench
    modulab sweep
    modulab representations
    modulab operators

## Representations

- stft
- filterbank_amfm
- parallel (wraps both, fusion = average or crossover)

## What this is NOT

No AI, no neural training, no ML. Pure physics + numerics.
