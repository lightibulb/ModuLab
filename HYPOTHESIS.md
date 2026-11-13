# Falsifiable Hypothesis

For narrowband AM/FM decomposition (BW/fc < 0.25), an AM/FM-based representation
supports speech transformations with equal-or-better reconstruction quality than
STFT at comparable FLOP cost, and provides explicit phase information that STFT
requires a separate predictor for.

## Acceptance criteria

1. AM/FM roundtrip SI-SDR within 3 dB of STFT on the synthetic corpus.
2. Parallel representation (AM/FM + STFT) at crossover fusion beats the best
   single representation by >= 1 dB SI-SDR at < 15% extra FLOPs.
3. Narrowband sweep shows SNR cliff around BW/fc = 0.25.

## Falsification

If (1) fails by more than 3 dB, or (2) never holds, the hypothesis is rejected.
