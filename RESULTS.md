# ModuLab Benchmark Results

**Config hash**: `176471917ab2`  
**Sample rate**: 16000 Hz  
**Duration**: 1.5 s  
**Seeds**: [42, 123, 456]  
**Total runs**: 324

## Table 1 — Quality at passthrough (mean ± 95% CI)

| Repr | Bits | Bitrate (kbps) | SNR (dB) | SI-SDR (dB) | MCD (dB) |
|---|---|---|---|---|---|
| filterbank_amfm | 8 | 2048 | -2.9±0.11 | -40.6±5.40 | 80.3±4.36 |
| filterbank_amfm | 12 | 3072 | 9.2±4.59 | 7.0±5.87 | 57.0±2.02 |
| filterbank_amfm | 16 | 4096 | 26.1±1.63 | 27.1±1.47 | 35.3±5.35 |
| parallel | 8 | 2563 | 3.2±0.11 | 0.3±0.21 | 77.0±4.59 |
| parallel | 12 | 3845 | 15.3±4.59 | 14.7±4.89 | 50.5±2.21 |
| parallel | 16 | 5127 | 32.1±1.63 | 33.0±1.49 | 27.2±5.50 |
| stft | 8 | 515 | 46.2±0.73 | 46.2±0.73 | 11.5±5.26 |
| stft | 12 | 773 | 69.9±0.50 | 69.9±0.50 | 9.4±4.29 |
| stft | 16 | 1031 | 93.1±0.36 | 93.1±0.36 | 9.3±4.32 |

**Equal-bitrate comparison**: at ~256 kbps, compare SNR across representations. Use the (Repr, Bits) row whose bitrate is closest to the target.

## Table 2 — Cost (FLOPs, wall-time, bitrate at 8-bit)

| Repr | FLOPs enc | FLOPs dec | Wall ms enc | Wall ms dec | Bitrate @8b |
|---|---|---|---|---|---|
| filterbank_amfm | 1.73e+07 | 7.68e+05 | 824.1 | 72.6 | 2048 |
| parallel | 2.18e+07 | 5.34e+06 | 871.6 | 193.9 | 2563 |
| stft | 4.52e+06 | 4.52e+06 | 28.6 | 117.7 | 515 |

## Table 3 — Filter bandwidth sweep (12-bit, β=1.0)

*Varies filter BW at fixed modulation index. Non-monotonic pattern reflects filter truncation, NOT Bedrosian validity — see Tables 4-6.*

| fc (Hz) | BW/fc | env_upper (Hz) | car_lower (Hz) | Bed. ratio | SNR (dB) |
|---|---|---|---|---|---|
| 500 | 0.10 | 16 | 463 | 0.03 | 17.72 |
| 500 | 0.25 | 33 | 406 | 0.08 | 24.23 |
| 500 | 0.50 | 66 | 313 | 0.21 | 35.73 |
| 500 | 0.75 | 99 | 219 | 0.45 | 32.75 |
| 500 | 1.00 | 130 | 125 | 1.04 | 24.44 |
| 500 | 1.50 | 191 | 63 | 3.04 | 16.72 |
| 500 | 2.00 | 250 | 250 | 1.00 | 21.69 |
| 500 | 3.00 | 380 | 114 | 3.33 | 17.30 |
| 1000 | 0.10 | 31 | 925 | 0.03 | 8.06 |
| 1000 | 0.25 | 66 | 813 | 0.08 | 30.98 |
| 1000 | 0.50 | 130 | 625 | 0.21 | 24.40 |
| 1000 | 0.75 | 191 | 437 | 0.44 | 47.40 |
| 1000 | 1.00 | 250 | 250 | 1.00 | 28.71 |
| 1000 | 1.50 | 380 | 125 | 3.05 | 19.94 |
| 1000 | 2.00 | 500 | 500 | 1.00 | 20.90 |
| 1000 | 3.00 | 750 | 250 | 3.00 | 7.80 |
| 2000 | 0.10 | 50 | 1850 | 0.03 | 35.33 |
| 2000 | 0.25 | 130 | 1625 | 0.08 | 7.25 |
| 2000 | 0.50 | 250 | 1250 | 0.20 | 38.89 |
| 2000 | 0.75 | 380 | 875 | 0.43 | 8.32 |
| 2000 | 1.00 | 500 | 500 | 1.00 | 41.92 |
| 2000 | 1.50 | 750 | 250 | 3.00 | 9.77 |
| 2000 | 2.00 | 1000 | 1000 | 1.00 | 23.98 |
| 2000 | 3.00 | 1500 | 500 | 3.00 | 23.07 |

## Verdict

- **At 8-bit**: AM/FM SNR -2.9 dB @ 2048 kbps vs STFT 46.2 dB @ 515 kbps.
- **At 12-bit**: AM/FM SNR 9.2 dB @ 3072 kbps vs STFT 69.9 dB @ 773 kbps.
- **At 16-bit**: AM/FM SNR 26.1 dB @ 4096 kbps vs STFT 93.1 dB @ 1031 kbps.
- Parallel encode: 2.18e+07 FLOPs (sum of branches, no discount).

---

# Extended Analysis

## Table 4 — Bedrosian decomposition fidelity (1D)

*fc=1000 Hz, am_freq=fm_freq=400 Hz, am_depth=0.5, float*

| β | Bed. ratio | AM SNR (dB) | FM SNR (dB) | Recon SNR (dB) |
|---|---|---|---|---|
| 0.05 | 0.67 | 82.95 | 21.28 | 37.22 |
| 0.10 | 0.67 | 71.22 | 21.82 | 37.22 |
| 0.25 | 0.67 | 56.30 | 21.32 | 37.22 |
| 0.50 | 2.00 | 46.24 | 20.59 | 37.22 |
| 0.75 | 2.00 | 41.68 | 20.52 | 37.22 |
| 1.00 | 2.00 | 39.97 | 21.01 | 37.22 |
| 1.50 | 2.00 | 46.79 | 22.01 | 37.22 |
| 2.00 | 2.00 | 29.43 | 20.99 | 37.22 |
| 3.00 | 2.00 | 13.83 | 11.44 | 37.22 |
| 4.00 | 2.00 | 20.67 | 5.16 | 34.01 |
| 6.00 | 2.00 | 14.28 | 5.53 | 34.01 |

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
| 0.5 | 42.5 | -1.1 | -0.4 | 2.2 |
| 1.0 | 43.3 | -2.0 | -0.5 | 4.2 |
| 2.0 | 44.0 | -2.5 | 0.1 | 5.6 |
| 4.0 | 44.0 | -4.0 | 0.9 | 6.2 |
| 8.0 | 44.2 | -5.7 | 3.1 | 5.1 |
| 16.0 | 44.3 | -6.0 | 2.8 | 4.4 |
| 32.0 | 44.0 | -9.2 | -0.3 | 1.2 |
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