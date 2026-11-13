modulab/
├── README.md                      # Pitch + hypothesis + results table + 3-command demo
├── HYPOTHESIS.md                  # The falsifiable claim, restated precisely
├── RESULTS.md                     # Auto-generated from last benchmark run
├── LICENSE                        # Apache-2.0
├── modulab_config.yaml            # Single source of truth (reproducibility)
│
├── docs/
│   ├── theory/
│   │   ├── narrowband_condition.md     # Derivation: BW/fc, Bedrosian, Teager-Kaiser
│   │   ├── amfm_uniqueness.md          # Bedrosian theorem: when is AM/FM unique?
│   │   └── error_analysis.md           # Where AM/FM decomposition fails, analytically
│   ├── adr/
│   │   ├── ADR-001-stft-baseline.md    # Why STFT with these params as baseline
│   │   ├── ADR-002-heterodyne.md       # Heterodyne vs complex bandpass
│   │   ├── ADR-003-metric-set.md       # Why SI-SDR+MCD+PESQ (not just SNR)
│   │   └── ADR-004-dataset-curation.md # Speech clips + synthetic chirps
│   ├── diagrams/
│   └── reports/                        # Auto-generated HTML + PDF per run
│
├── modulab/                            # Python package
│   ├── pyproject.toml
│   ├── modulab/
│   │   ├── config.py                   # Loads YAML → typed dataclass
│   │   ├── representations/            # Pluggable: encode/decode
│   │   │   ├── base.py                 # Abstract Representation
│   │   │   ├── stft.py
│   │   │   ├── filterbank_amfm.py
│   │   │   ├── gammatone_amfm.py
│   │   │   ├── cqt.py                  # Comparison only
│   │   │   └── mel.py                  # Comparison only
│   │   ├── operators/                  # Pluggable: transform-in-domain
│   │   │   ├── base.py
│   │   │   ├── passthrough.py          # Upper bound on reconstruction quality
│   │   │   ├── time_stretch.py
│   │   │   ├── pitch_shift.py
│   │   │   ├── noise_gate.py
│   │   │   └── formant_shift.py        # Stretch-specific
│   │   ├── metrics/
│   │   │   ├── quality.py              # SI-SDR, SNR, MCD, PESQ, STOI
│   │   │   ├── cost.py                 # FLOPs, wall-time, peak RAM
│   │   │   ├── sparsity.py             # Nonzero fraction, entropy
│   │   │   └── flops.py                # Analytic + profiled FLOP counter
│   │   ├── harness/
│   │   │   ├── runner.py               # Orchestrates the matrix
│   │   │   ├── matrix.py               # (repr × op × clip × seed)
│   │   │   └── report.py               # JSON → markdown + plots
│   │   ├── viz/
│   │   │   ├── spectrogram.py
│   │   │   ├── amfm_trace.py           # AM/FM vs time per band
│   │   │   ├── error_maps.py           # Reconstruction error heatmap
│   │   │   └── cost_curves.py          # FLOPs vs. BW/fc
│   │   └── data/
│   │       ├── loaders.py              # speech clips + synthetic chirps
│   │       └── synth.py                # Chirp, AM tone, FM tone, test signals
│   ├── tests/
│   │   ├── test_representations.py     # Roundtrip fidelity bounds
│   │   ├── test_operators.py           # Op correctness (known-input invariants)
│   │   ├── test_metrics.py             # Known-value sanity tests
│   │   └── test_harness.py             # Matrix completeness, reproducibility
│   └── notebooks/
│       ├── 01_narrowband_theory.ipynb  # Interactive derivation + plots
│       ├── 02_amfm_vs_stft.ipynb       # First manual comparison
│       └── 03_failure_modes.ipynb      # Where AM/FM breaks
│
├── benchmarks/                         # CLI entrypoints
│   ├── run_benchmark.py                # Full matrix
│   ├── run_analysis.py                 # Narrowband-ratio sweep
│   └── reproduce.sh                    # One-command reproduction
│
├── data/
│   ├── speech/                         # git-lfs: ~20 clips, ~30 s total
│   │   ├── librispeech_dev_clean/
│   │   ├── vctk_subset/
│   │   └── manifest.json
│   └── synthetic/                      # Generated, versioned by seed
│
├── results/
│   ├── runs/                           # JSON: one per benchmark run (with config hash)
│   ├── figures/                        # PNG/SVG
│   └── latest -> runs/YYYY-MM-DD_hhmmss # Symlink for report
│
├── scripts/
│   ├── gen_synthetic.py                # Deterministic test signals
│   ├── profile_flops.py                # Static FLOP analysis
│   └── gen_report.py                   # results/*.json → RESULTS.md + plots
│
├── docker/
│   └── Dockerfile                      # Pinned: python, numpy, scipy, librosa, pypesq
│
├── .pre-commit-config.yaml
└── .github/workflows/
    ├── test.yml                        # pytest + ruff + mypy
    ├── benchmark_smoke.yml             # Tiny matrix on synthetic (CI <2 min)
    └── docs.yml
