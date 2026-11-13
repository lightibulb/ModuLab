from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np


def _fmt(v, nd=2):
    if isinstance(v, float):
        return "n/a" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def generate_report(results, sweep, out_dir, config_path):
    L = []
    L.append("# ModuLab Benchmark Results")
    L.append("")
    L.append(f"**Config hash**: `{results['config_hash']}`  ")
    L.append(f"**Sample rate**: {results['fs']} Hz  ")
    L.append(f"**Duration**: {results['duration_s']} s  ")
    L.append(f"**Total runs**: {len(results['runs'])}")
    L.append("")

    L.append("## Table 1 - Quality at passthrough")
    L.append("")
    L.append("| Representation | Bits | SNR (dB) | SI-SDR (dB) | MCD (dB) |")
    L.append("|---|---|---|---|---|")
    agg = defaultdict(lambda: defaultdict(list))
    for r in results["runs"]:
        if "error" in r or r.get("op") != "passthrough":
            continue
        key = (r["repr"], r["bits"])
        for m, v in r["metrics"].items():
            agg[key][m].append(v)
    for (rn, b) in sorted(agg.keys()):
        m = {k: float(np.nanmean(v)) for k, v in agg[(rn, b)].items()}
        L.append(f"| {rn} | {b} | {_fmt(m.get('snr_db'))} | "
                 f"{_fmt(m.get('si_sdr_db'))} | {_fmt(m.get('mcd_db'))} |")
    L.append("")

    L.append("## Table 2 - Computational cost")
    L.append("")
    L.append("| Representation | FLOPs encode | FLOPs decode | Wall ms enc | Wall ms dec |")
    L.append("|---|---|---|---|---|")
    agg2 = defaultdict(lambda: defaultdict(list))
    for r in results["runs"]:
        if "error" in r:
            continue
        agg2[r["repr"]]["flops_encode"].append(r["flops_encode"])
        agg2[r["repr"]]["flops_decode"].append(r["flops_decode"])
        agg2[r["repr"]]["wall_ms_encode"].append(r["wall_ms_encode"])
        agg2[r["repr"]]["wall_ms_decode"].append(r["wall_ms_decode"])
    for rn in sorted(agg2.keys()):
        m = {k: float(np.mean(v)) for k, v in agg2[rn].items()}
        L.append(f"| {rn} | {m['flops_encode']:.2e} | {m['flops_decode']:.2e} | "
                 f"{_fmt(m['wall_ms_encode'])} | {_fmt(m['wall_ms_decode'])} |")
    L.append("")

    if sweep and "sweep" in sweep:
        L.append("## Table 3 - Narrowband sweep (8-bit, single band)")
        L.append("")
        L.append("| fc (Hz) | BW/fc | BW (Hz) | SNR (dB) |")
        L.append("|---|---|---|---|")
        for row in sweep["sweep"]:
            L.append(f"| {row['fc_hz']:.0f} | {row['bw_ratio']:.2f} | "
                     f"{row['bw_hz']:.0f} | {row['snr_db']:.2f} |")
        L.append("")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            for fc in sorted(set(r["fc_hz"] for r in sweep["sweep"])):
                pts = sorted([r for r in sweep["sweep"] if r["fc_hz"] == fc],
                             key=lambda z: z["bw_ratio"])
                ax.plot([p["bw_ratio"] for p in pts], [p["snr_db"] for p in pts],
                        marker="o", label=f"fc={fc:.0f} Hz")
            ax.axvline(0.25, linestyle="--", color="gray", alpha=0.6,
                       label="Narrowband limit (BW/fc=0.25)")
            ax.set_xlabel("BW / fc")
            ax.set_ylabel("Reconstruction SNR (dB)")
            ax.set_title("AM/FM Reconstruction Quality vs Narrowband Ratio")
            ax.grid(True, alpha=0.3)
            ax.legend()
            out_dir.mkdir(parents=True, exist_ok=True)
            fig_path = out_dir / "narrowband_sweep.png"
            fig.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            L.append(f"![Narrowband sweep]({fig_path.name})")
            L.append("")
        except Exception as e:
            L.append(f"(Plot failed: {e})")
            L.append("")

    L.append("## Verdict")
    L.append("")
    try:
        if ("filterbank_amfm", 8) in agg and ("stft", 8) in agg:
            a_snr = float(np.nanmean(agg[("filterbank_amfm", 8)]["snr_db"]))
            s_snr = float(np.nanmean(agg[("stft", 8)]["snr_db"]))
            a_f = float(np.mean(agg2["filterbank_amfm"]["flops_encode"]))
            s_f = float(np.mean(agg2["stft"]["flops_encode"]))
            L.append(f"At 8-bit, AM/FM SNR {a_snr:.1f} dB vs STFT {s_snr:.1f} dB; "
                     f"AM/FM encode {a_f:.2e} FLOPs vs STFT {s_f:.2e} FLOPs.")
        if "parallel" in agg2:
            par_f = float(np.mean(agg2["parallel"]["flops_encode"]))
            L.append(f"Parallel encode costs {par_f:.2e} FLOPs "
                     f"(sum of branches, no discount).")
        if sweep and "sweep" in sweep:
            nb = [r for r in sweep["sweep"] if r["bw_ratio"] <= 0.25]
            wb = [r for r in sweep["sweep"] if r["bw_ratio"] >= 0.5]
            if nb and wb:
                L.append(f"Narrowband (BW/fc<=0.25) SNR {np.mean([r['snr_db'] for r in nb]):.1f} dB "
                         f"vs wideband (BW/fc>=0.5) {np.mean([r['snr_db'] for r in wb]):.1f} dB.")
    except Exception as e:
        L.append(f"(Verdict failed: {e})")
    L.append("")

    text = "\n".join(L)
    Path("RESULTS.md").write_text(text)
    return text
