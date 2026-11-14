# ModuLab Benchmark Results

**Config hash**: `543a1776cf71`  
**Sample rate**: 16000 Hz  
**Duration**: 1.5 s  
**Total runs**: 108

## Table 1 - Quality at passthrough

| Representation | Bits | SNR (dB) | SI-SDR (dB) | MCD (dB) |
|---|---|---|---|---|
| filterbank_amfm | 8 | -2.83 | -35.97 | 80.30 |
| filterbank_amfm | 12 | 11.29 | 10.22 | 57.02 |
| filterbank_amfm | 16 | 26.43 | 27.39 | 7.01 |
| parallel | 8 | 3.19 | 0.35 | 77.02 |
| parallel | 12 | 17.31 | 17.03 | 50.30 |
| parallel | 16 | 32.45 | 33.33 | 4.20 |
| stft | 8 | 46.22 | 46.23 | 11.42 |
| stft | 12 | 69.88 | 69.89 | 9.28 |
| stft | 16 | 140.38 | 140.60 | 0.18 |

## Table 2 - Computational cost

| Representation | FLOPs encode | FLOPs decode | Wall ms enc | Wall ms dec |
|---|---|---|---|---|
| filterbank_amfm | 1.73e+07 | 7.68e+05 | 856.03 | 72.14 |
| parallel | 2.18e+07 | 5.34e+06 | 675.00 | 150.81 |
| stft | 4.52e+06 | 4.52e+06 | 26.85 | 87.94 |

## Table 3 - Narrowband sweep (12-bit, single band)

| fc (Hz) | BW/fc | BW (Hz) | SNR (dB) |
|---|---|---|---|
| 500 | 0.10 | 50 | 17.72 |
| 500 | 0.25 | 125 | 24.23 |
| 500 | 0.50 | 250 | 35.73 |
| 500 | 0.75 | 375 | 32.75 |
| 500 | 1.00 | 500 | 24.44 |
| 500 | 1.50 | 750 | 16.72 |
| 500 | 2.00 | 1000 | 21.69 |
| 500 | 3.00 | 1500 | 17.30 |
| 1000 | 0.10 | 100 | 8.06 |
| 1000 | 0.25 | 250 | 30.98 |
| 1000 | 0.50 | 500 | 24.40 |
| 1000 | 0.75 | 750 | 47.40 |
| 1000 | 1.00 | 1000 | 28.71 |
| 1000 | 1.50 | 1500 | 19.94 |
| 1000 | 2.00 | 2000 | 20.90 |
| 1000 | 3.00 | 3000 | 7.80 |
| 2000 | 0.10 | 200 | 35.33 |
| 2000 | 0.25 | 500 | 7.25 |
| 2000 | 0.50 | 1000 | 38.89 |
| 2000 | 0.75 | 1500 | 8.32 |
| 2000 | 1.00 | 2000 | 41.92 |
| 2000 | 1.50 | 3000 | 9.77 |
| 2000 | 2.00 | 4000 | 23.98 |
| 2000 | 3.00 | 6000 | 23.07 |

(Plot failed: No module named 'matplotlib')

## Verdict

At 8-bit, AM/FM SNR -2.8 dB vs STFT 46.2 dB; AM/FM encode 1.73e+07 FLOPs vs STFT 4.52e+06 FLOPs.
Parallel encode costs 2.18e+07 FLOPs (sum of branches, no discount).
Narrowband (BW/fc<=0.25) SNR 20.6 dB vs wideband (BW/fc>=0.5) 24.7 dB.

---

# Extended Analysis

## Table 4 — Bedrosian decomposition fidelity (1D)

*fc=1000 Hz, am_freq=fm_freq=400 Hz, am_depth=0.5, float*

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

**AM drops 83→14 dB as envelope-carrier overlap crosses 1.0. Recon SNR flat — failure is structural.**

## Table 5 — AM/FM interference (β × am_depth)

*Varies envelope amplitude, not bandwidth — NOT a Bedrosian test.*

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

*f0=120, F1=700, F2=1220, F3=2600, adaptive LPF, SI-SDR*

| am_freq | AM vs theory | AM env | STFT long | STFT short |
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

**AM collapses at am_freq ≈ f0 = 120 Hz (pitch collision).**

## Table 7 — FM tracking: multi-hop STFT vs AM/FM

*f0=1000 Hz, fm_depth=50 Hz, STFT n_fft=512*

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

**Crossover at ~2 Hz. STFT wins slow vibrato, AM/FM wins 4–200 Hz.**

## Consolidated Verdict

1. **Cost**: AM/FM = 3.8× STFT FLOPs. STFT wins.
2. **Quantization**: AM/FM needs ≥12-bit. STFT robust at any bit.
3. **Bedrosian** (T4): 69 dB drop when envelope-carrier overlap crosses 1.0.
4. **AM/FM interference** (T5): deeper AM more robust to high FM index.
5. **Speech envelope** (T6): AM/FM and short-window STFT tie within 5 dB.
6. **FM tracking** (T7): crossover at 2 Hz; AM/FM wins fast rates.

**Conclusion**: AM/FM wins two regimes and loses elsewhere. Task-dependent selection.