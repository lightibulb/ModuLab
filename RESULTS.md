
---

# Extended Analysis (sweeps from separate experiments)

## Table 4 — Bedrosian decomposition fidelity (1D)

fc=1000 Hz, am_freq=fm_freq=400 Hz, am_depth=0.5, float

| β | Bed. ratio | AM SNR (dB) | FM SNR (dB) | Recon SNR (dB) |
|---|---|---|---|---|
| 0.05 | 0.61 | 83.0 | 21.3 | 37.2 |
| 0.10 | 0.62 | 71.2 | 21.8 | 37.2 |
| 0.25 | 0.67 | 56.3 | 21.3 | 37.2 |
| 0.50 | 0.75 | 46.2 | 20.6 | 37.2 |
| 0.75 | 0.86 | 41.7 | 20.5 | 37.2 |
| 1.00 | 1.00 | 40.0 | 21.0 | 37.2 |
| 1.50 | 1.50 | 46.8 | 22.0 | 37.2 |
| 2.00 | 3.00 | 29.4 | 21.0 | 37.2 |
| 3.00 | inf | 13.8 | 11.4 | 37.2 |
| 4.00 | inf | 20.7 | 5.2 | 34.0 |
| 6.00 | inf | 14.3 | 5.5 | 34.0 |

**AM decomposition drops 69 dB (83 → 14) as envelope-carrier overlap crosses 1.0. Reconstruction SNR stays flat — the failure is structural, not numerical.**

## Table 6 — Speech-like envelope tracking (fair comparison)

Multi-formant (f0=120, F1=700, F2=1220, F3=2600), adaptive LPF

| am_freq | LPF | AM_Thy | AM_env | STFT_lng | STFT_sht |
|---|---|---|---|---|---|
| 1.0 | 30 | 84.2 | 67.9 | 65.2 | 65.3 |
| 2.0 | 30 | 95.7 | 69.7 | 63.8 | 66.5 |
| 4.0 | 30 | 99.2 | 70.5 | 55.2 | 65.2 |
| 8.0 | 30 | 97.6 | 70.5 | 44.3 | 63.9 |
| 16.0 | 30 | 98.0 | 69.2 | 43.3 | 62.2 |
| 32.0 | 48 | 98.7 | 60.4 | 40.3 | 58.6 |
| 64.0 | 96 | 97.9 | 18.3 | -2.4 | 15.8 |
| 128.0 | 192 | 41.2 | 7.1 | -65.0 | 4.1 |
| 256.0 | 384 | 36.0 | 5.4 | -76.6 | -14.3 |
| 512.0 | 500 | 26.5 | -2.2 | -69.4 | -60.9 |
| 700.0 | 500 | 18.7 | -19.5 | -89.5 | -105.6 |

**AM_Thy collapses at am_freq ≈ f0 = 120 Hz (pitch collision). AM_env vs short-window STFT tie within 5 dB at speech rates.**

## Table 7 — FM tracking: multi-hop STFT vs AM/FM

f0=1000 Hz, fm_depth=50 Hz, STFT n_fft=512

| fm_freq | AM/FM | STFT hop=32 | STFT hop=128 | STFT hop=256 |
|---|---|---|---|---|
| 0.5 | 42.5 | 60.5 | 60.6 | -2.7 |
| 1.0 | 43.3 | 57.5 | 55.6 | -2.1 |
| 2.0 | 44.0 | 40.5 | 40.4 | -2.3 |
| 4.0 | 44.0 | 29.2 | 29.5 | -1.7 |
| 8.0 | 44.2 | 18.3 | 20.6 | -1.2 |
| 16.0 | 44.3 | 10.8 | 13.2 | -1.4 |
| 32.0 | 44.0 | 6.9 | 2.9 | -0.0 |
| 64.0 | 43.8 | -0.1 | 0.0 | -0.0 |
| 128.0 | 43.5 | -0.0 | -0.0 | -0.0 |
| 200.0 | 43.8 | -0.0 | 0.0 | -0.0 |

**Crossover at ~2 Hz. STFT wins slow vibrato (coherent window), AM/FM wins 4–200 Hz by 15–40 dB (sample-rate). STFT hop=256 fails at all rates — 2× overlap is below the 4× coherence threshold.**

## Consolidated Verdict

Across all experiments:

1. **Cost**: AM/FM = 3.8× STFT FLOPs, 36× wall time. STFT wins.
2. **Quantization**: AM/FM needs ≥12-bit FM. STFT robust at any bit.
3. **Bedrosian**: AM/FM decomposition collapses by 69 dB when envelope-carrier overlap exceeds 1.0. Reconstruction SNR stays flat (>37 dB) — the failure is structural, invisible in recon metrics.
4. **Envelope at speech rates**: AM/FM and short-window STFT are within ~5 dB. Neither dominates.
5. **FM tracking**: STFT wins ≤2 Hz (+14–18 dB), AM/FM wins ≥4 Hz (+15–40 dB). Crossover at 2 Hz. STFT requires hop ≤ n_fft/4 for phase coherence — a constraint AM/FM doesn't have.
6. **Pitch collision**: both fail when am_freq > f0. Fundamental limit of envelope-based representations.

**Conclusion**: AM/FM is not faster or universally better than STFT, but it wins two specific regimes (envelope at speech rates, fast FM tracking above 2 Hz). The correct engineering answer is task-dependent selection, not universal substitution. The portfolio value is this regime map — quantified, empirical, honest.
## Table 4 — Bedrosian decomposition fidelity (1D)

*fc=1000 Hz, am_freq=fm_freq=400 Hz, am_depth=0.5, float precision*

| β | Bed. ratio | AM SNR (dB) | FM SNR (dB) | Recon SNR (dB) |
|---|---|---|---|---|
| 0.05 | 0.61 | 82.95 | 21.28 | 37.22 |
| 0.10 | 0.62 | 71.22 | 21.82 | 37.22 |
| 0.25 | 0.67 | 56.30 | 21.32 | 37.22 |
| 0.50 | 0.75 | 46.24 | 20.59 | 37.22 |
| 0.75 | 0.86 | 41.68 | 20.52 | 37.22 |
| 1.00 | 1.00 | 39.97 | 21.01 | 37.22 |
| 1.50 | 1.50 | 46.79 | 22.01 | 37.22 |
| 2.00 | 3.00 | 29.43 | 20.99 | 37.22 |
| 3.00 | inf | 13.83 | 11.44 | 37.22 |
| 4.00 | inf | 20.67 | 5.16 | 34.01 |
| 6.00 | inf | 14.28 | 5.53 | 34.01 |

**AM SNR drops 83 → 14 dB (69 dB) as envelope-carrier overlap crosses 1.0. Reconstruction SNR stays flat — the failure is structural, invisible in recon metrics.**

## Table 5 — AM/FM interference diagram (β × am_depth)

*fc=1000 Hz, am_freq=fm_freq=400 Hz. This varies envelope amplitude, not bandwidth, so it is NOT a Bedrosian test. It shows the resonance between AM depth and FM index: deeper AM makes AM estimation more robust to phase corruption from high FM index.*

| β \ am_depth | 0.10 | 0.30 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|
| 0.05 | 74.7 | 73.6 | 73.4 | 73.3 | 73.3 |
| 0.25 | 60.7 | 48.0 | 46.8 | 46.3 | 46.0 |
| 0.50 | 36.8 | 40.2 | 36.7 | 35.5 | 34.9 |
| 0.75 | 22.8 | 41.5 | 32.2 | 30.0 | 28.9 |
| 1.00 | 14.2 | 40.6 | 30.7 | 26.9 | 25.1 |
| 1.50 | 2.9 | 18.6 | 38.9 | 24.9 | 21.2 |
| 2.00 | -4.6 | 8.9 | 20.7 | 29.0 | 20.5 |
| 3.00 | -12.9 | -1.1 | 6.6 | 14.4 | 21.7 |

## Table 6 — Speech-like envelope tracking

*Multi-formant (f0=120, F1=700, F2=1220, F3=2600), adaptive LPF, SI-SDR*

| am_freq (Hz) | AM vs theory | AM env vs truth | STFT long | STFT short |
|---|---|---|---|---|
| 1.0 | 84.2 | 67.9 | 65.2 | 65.3 |
| 2.0 | 95.7 | 69.7 | 63.8 | 66.5 |
| 4.0 | 99.2 | 70.5 | 55.2 | 65.2 |
| 8.0 | 97.6 | 70.5 | 44.3 | 63.9 |
| 16.0 | 98.0 | 69.2 | 43.3 | 62.2 |
| 32.0 | 98.7 | 60.4 | 40.3 | 58.6 |
| 64.0 | 97.9 | 18.3 | -2.4 | 15.8 |
| 128.0 | 41.2 | 7.1 | -65.0 | 4.1 |
| 256.0 | 36.0 | 5.4 | -76.6 | -14.3 |
| 512.0 | 26.5 | -2.2 | -69.4 | -60.9 |
| 700.0 | 18.7 | -19.5 | -89.5 | -105.6 |

**AM_Thy collapses at am_freq ≈ f0 = 120 Hz (pitch collision). AM/FM and short-window STFT tie within 5 dB at speech rates.**

## Table 7 — FM tracking: multi-hop STFT vs AM/FM

*f0=1000 Hz, fm_depth=50 Hz, STFT n_fft=512*

| fm_freq (Hz) | AM/FM | STFT hop=32 | STFT hop=128 | STFT hop=256 |
|---|---|---|---|---|
| 0.5 | 42.5 | 60.5 | 60.6 | -2.7 |
| 1.0 | 43.3 | 57.5 | 55.6 | -2.1 |
| 2.0 | 44.0 | 40.5 | 40.4 | -2.3 |
| 4.0 | 44.0 | 29.2 | 29.5 | -1.7 |
| 8.0 | 44.2 | 18.3 | 20.6 | -1.2 |
| 16.0 | 44.3 | 10.8 | 13.2 | -1.4 |
| 32.0 | 44.0 | 6.9 | 2.9 | -0.0 |
| 64.0 | 43.8 | -0.1 | 0.0 | -0.0 |
| 128.0 | 43.5 | -0.0 | -0.0 | -0.0 |
| 200.0 | 43.8 | -0.0 | 0.0 | -0.0 |

**Crossover at ~2 Hz. STFT wins slow vibrato (long coherent window). AM/FM wins 4–200 Hz by 15–40 dB (sample-rate tracking). STFT hop=256 fails at all rates — insufficient overlap for phase coherence.**

## Consolidated Verdict

Across four independent experiments:

1. **Cost**: AM/FM = 3.8× STFT FLOPs, ~40× wall time. STFT wins.
2. **Quantization**: AM/FM unusable at 8-bit; needs ≥12-bit. STFT robust at any bit depth.
3. **Bedrosian validity** (Table 4): AM decomposition drops 83 → 14 dB as envelope-carrier overlap crosses 1.0. Reconstruction SNR flat — failure is structural, hidden by recon metrics.
4. **AM/FM interference** (Table 5): deeper envelope modulation makes AM estimation more robust to high FM index. Diagonal ridge in (β, am_depth). Not Bedrosian — resonance effect.
5. **Speech envelope tracking** (Table 6): AM/FM and short-window STFT within 5 dB at speech rates. Both fail above am_freq ≈ f0 = 120 Hz (pitch collision) — fundamental limit of envelope representations.
6. **FM tracking** (Table 7): STFT wins ≤2 Hz by 14–18 dB. AM/FM wins 4–200 Hz by 15–40 dB. Crossover at 2 Hz. STFT requires hop ≤ n_fft/4 for phase coherence.

**Conclusion**: AM/FM is not faster or universally better than STFT. It wins two regimes (envelope at speech rates, fast FM tracking) and loses elsewhere (cost, quantization, slow-vibrato FM). The correct engineering answer is task-dependent selection. This regime map is the empirical result.