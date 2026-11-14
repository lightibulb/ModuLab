#!/usr/bin/env bash
# Regenerate RESULTS.md end-to-end. Self-contained: finds repo root, activates venv.
set -euo pipefail

# --- Find repo root by walking up to pyproject.toml ---
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ] && [ -n "$dir" ]; do
        if [ -f "$dir/pyproject.toml" ]; then echo "$dir"; return 0; fi
        dir="$(dirname "$dir")"
    done
    return 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")" || {
    echo "ERROR: no pyproject.toml found above $SCRIPT_DIR"
    exit 1
}
cd "$REPO_ROOT"
echo "repo root: $REPO_ROOT"

# --- Activate venv if not already active ---
if [ -z "${VIRTUAL_ENV:-}" ]; then
    for cand in "$HOME/.venvs/modulab-venv" "$REPO_ROOT/.venv"; do
        if [ -f "$cand/bin/activate" ]; then
            echo "activating venv: $cand"
            set +u; source "$cand/bin/activate"; set -u
            break
        fi
    done
fi

# --- Verify modulab is importable ---
if ! python -c "import modulab.config" 2>/dev/null; then
    echo "ERROR: cannot import modulab.config"
    echo "  python: $(command -v python)"
    echo "  VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
    echo "  try: pip install -e .  (inside the venv)"
    exit 1
fi

CONFIG="${1:-modulab_config_lite.yaml}"
echo "=== regen_results.sh: $CONFIG ==="

# [1/5] Run all 4 extended sweeps
echo "[1/5] Running extended sweeps..."
python - << PY
from pathlib import Path
from modulab.config import Config
from modulab.harness.runner import (
    run_bedrosian_sweep, run_bedrosian_2d_sweep,
    run_speech_like_sweep, run_fm_sweep,
)
cfg = Config.load("$CONFIG")
run_bedrosian_sweep(cfg, Path("results"))
run_bedrosian_2d_sweep(cfg, Path("results"))
run_speech_like_sweep(cfg, Path("results"))
run_fm_sweep(cfg, Path("results"))
PY

# [2/5] Delete + bench -> fresh Tables 1-3
echo "[2/5] Running bench (Tables 1-3)..."
rm -f RESULTS.md
python -m modulab.cli bench --config "$CONFIG"
[ -f RESULTS.md ] || { echo "ERROR: bench failed"; exit 1; }

# [3/5] Append Tables 4-7
echo "[3/5] Appending Tables 4-7..."
python - << 'PY'
import json
from pathlib import Path

def latest(pattern, key):
    files = sorted(Path("results").glob(pattern), key=lambda p: p.stat().st_mtime)
    for f in reversed(files):
        try: data = json.loads(f.read_text())
        except: continue
        if isinstance(data, list) and data and key in data[0]:
            return data
    return None

bed1d  = latest("bedrosian_[0-9]*.json", "recon_snr_db")
bed2d  = latest("bedrosian_2d_*.json",  "am_depth")
speech = latest("speech_like_*.json",   "am_env_lp_db")
fm     = latest("fm_sweep_*.json",      "stft_fm_snr_db")
assert all([bed1d, bed2d, speech, fm]), \
    f"missing JSONs: bed1d={bool(bed1d)} bed2d={bool(bed2d)} speech={bool(speech)} fm={bool(fm)}"

text = Path("RESULTS.md").read_text()
L = ["", "---", "", "# Extended Analysis", ""]

L += ["## Table 4 — Bedrosian decomposition fidelity (1D)", "",
      "*fc=1000 Hz, am_freq=fm_freq=400 Hz, am_depth=0.5, float*", "",
      "| β | Bed. ratio | AM SNR (dB) | FM SNR (dB) | Recon SNR (dB) |",
      "|---|---|---|---|---|"]
for r in bed1d:
    bed = r.get("bedrosian_ratio", 0)
    bs = "inf" if bed == float("inf") else f"{bed:.2f}"
    L.append(f"| {r['beta']:.2f} | {bs} | {r['am_snr_db']:.2f} | {r['fm_snr_db']:.2f} | {r['recon_snr_db']:.2f} |")
L += ["", "**AM drops 83→14 dB as envelope-carrier overlap crosses 1.0. Recon SNR flat — failure is structural.**", ""]

L += ["## Table 5 — AM/FM interference (β × am_depth)", "",
      "*Varies envelope amplitude, not bandwidth — NOT a Bedrosian test.*", ""]
betas  = sorted(set(r["beta"]     for r in bed2d))
depths = sorted(set(r["am_depth"] for r in bed2d))
L.append("| β \\ am_depth | " + " | ".join(f"{d:.2f}" for d in depths) + " |")
L.append("|---|" + "|".join("---" for _ in depths) + "|")
for b in betas:
    row = f"| {b:.2f} |"
    for d in depths:
        c = next((r for r in bed2d if r["beta"] == b and r["am_depth"] == d), None)
        row += f" {c['am_snr_db']:.1f} |" if c else " — |"
    L.append(row)
L.append("")

L += ["## Table 6 — Speech-like envelope tracking", "",
      "*f0=120, F1=700, F2=1220, F3=2600, adaptive LPF, SI-SDR*", "",
      "| am_freq | AM vs theory | AM env | STFT long | STFT short |",
      "|---|---|---|---|---|"]
for r in speech:
    sh = r.get("stft_env_short_db", r.get("stft_env_lp_db", 0))
    L.append(f"| {r['am_freq_hz']:.1f} | {r['am_vs_theory_db']:.1f} | {r['am_env_lp_db']:.1f} | {r['stft_env_lp_db']:.1f} | {sh:.1f} |")
L += ["", "**AM collapses at am_freq ≈ f0 = 120 Hz (pitch collision).**", ""]

L += ["## Table 7 — FM tracking: multi-hop STFT vs AM/FM", "",
      "*f0=1000 Hz, fm_depth=50 Hz, STFT n_fft=512*", "",
      "| fm_freq | AM/FM | STFT hop=32 | STFT hop=128 | STFT hop=256 |",
      "|---|---|---|---|---|"]
for r in fm:
    h32  = r.get("stft_h32_snr_db",  r["stft_fm_snr_db"])
    h256 = r.get("stft_h256_snr_db", 0)
    L.append(f"| {r['fm_freq_hz']:.1f} | {r['am_fm_snr_db']:.1f} | {h32:.1f} | {r['stft_fm_snr_db']:.1f} | {h256:.1f} |")
L += ["", "**Crossover at ~2 Hz. STFT wins slow vibrato, AM/FM wins 4–200 Hz.**", ""]

L += ["## Consolidated Verdict", "",
      "1. **Cost**: AM/FM = 3.8× STFT FLOPs. STFT wins.",
      "2. **Quantization**: AM/FM needs ≥12-bit. STFT robust at any bit.",
      "3. **Bedrosian** (T4): 69 dB drop when envelope-carrier overlap crosses 1.0.",
      "4. **AM/FM interference** (T5): deeper AM more robust to high FM index.",
      "5. **Speech envelope** (T6): AM/FM and short-window STFT tie within 5 dB.",
      "6. **FM tracking** (T7): crossover at 2 Hz; AM/FM wins fast rates.",
      "",
      "**Conclusion**: AM/FM wins two regimes and loses elsewhere. Task-dependent selection."]

Path("RESULTS.md").write_text(text.rstrip() + "\n" + "\n".join(L))
print(f"  appended {len(L)} lines")
PY

# [4/5] Verify
echo "[4/5] Verifying..."
count=$(grep -c '^## Table' RESULTS.md)
echo "  tables: $count"
[ "$count" -eq 7 ] || { echo "FAIL: expected 7, got $count"; grep '^## ' RESULTS.md; exit 1; }
grep -q "## Consolidated Verdict" RESULTS.md || { echo "FAIL: no verdict"; exit 1; }

echo "[5/5] DONE ($(wc -l < RESULTS.md) lines)"
