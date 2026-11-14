from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as st


def _ci95(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if len(arr) == 1:
        return (float(arr[0]), 0.0)
    m = float(arr.mean())
    h = float(st.sem(arr) * st.t.ppf(0.975, len(arr) - 1))
    return (m, h)


def _fmt_ci(mean, ci):
    if np.isnan(mean):
        return "n/a"
    if ci <= 0:
        return f"{mean:.1f}"
    return f"{mean:.1f}±{ci:.2f}"


def _fmt_kbps(bps):
    if bps is None or (isinstance(bps, float) and np.isnan(bps)):
        return "n/a"
    return f"{bps / 1000.0:.0f}"


def generate_report(results, sweep, out_dir, config_path):
    L = []
    L.append("# ModuLab Benchmark Results")
    L.append("")
    L.append(f"**Config hash**: `{results['config_hash']}`  ")
    L.append(f"**Sample rate**: {results['fs']} Hz  ")
    L.append(f"**Duration**: {results['duration_s']} s  ")
    L.append(f"**Seeds**: {results.get('seeds', [42])}  ")
    L.append(f"**Total runs**: {len(results['runs'])}")
    L.append("")

    # ---- Table 1: quality with CI + bitrate ----
    L.append("## Table 1 — Quality at passthrough (mean ± 95% CI)")
    L.append("")
    L.append("| Repr | Bits | Bitrate (kbps) | SNR (dB) | SI-SDR (dB) | MCD (dB) |")
    L.append("|---|---|---|---|---|---|")
    agg = defaultdict(lambda: defaultdict(list))
    for r in results["runs"]:
        if "error" in r or r.get("op") != "passthrough":
            continue
        key = (r["repr"], r["bits"])
        agg[key]["snr_db"].append(r["metrics"].get("snr_db", float("nan")))
        agg[key]["si_sdr_db"].append(r["metrics"].get("si_sdr_db", float("nan")))
        agg[key]["mcd_db"].append(r["metrics"].get("mcd_db", float("nan")))
        agg[key]["bitrate_bps"].append(r.get("bitrate_bps", float("nan")))
    for (rn, b) in sorted(agg.keys()):
        snr_m, snr_ci = _ci95(agg[(rn, b)]["snr_db"])
        si_m, si_ci = _ci95(agg[(rn, b)]["si_sdr_db"])
        mc_m, mc_ci = _ci95(agg[(rn, b)]["mcd_db"])
        br_m, _ = _ci95(agg[(rn, b)]["bitrate_bps"])
        L.append(f"| {rn} | {b} | {_fmt_kbps(br_m)} | {_fmt_ci(snr_m, snr_ci)} | "
                 f"{_fmt_ci(si_m, si_ci)} | {_fmt_ci(mc_m, mc_ci)} |")
    L.append("")
    L.append("**Equal-bitrate comparison**: at ~256 kbps, compare SNR across representations. "
             "Use the (Repr, Bits) row whose bitrate is closest to the target.")
    L.append("")

    # ---- Table 2: cost + bitrate ----
    L.append("## Table 2 — Cost (FLOPs, wall-time, bitrate at 8-bit)")
    L.append("")
    L.append("| Repr | FLOPs enc | FLOPs dec | Wall ms enc | Wall ms dec | Bitrate @8b |")
    L.append("|---|---|---|---|---|---|")
    agg2 = defaultdict(lambda: defaultdict(list))
    for r in results["runs"]:
        if "error" in r:
            continue
        agg2[r["repr"]]["flops_encode"].append(r["flops_encode"])
        agg2[r["repr"]]["flops_decode"].append(r["flops_decode"])
        agg2[r["repr"]]["wall_ms_encode"].append(r["wall_ms_encode"])
        agg2[r["repr"]]["wall_ms_decode"].append(r["wall_ms_decode"])
        if r["bits"] == 8:
            agg2[r["repr"]]["bitrate_bps_8"].append(r.get("bitrate_bps", float("nan")))
    for rn in sorted(agg2.keys()):
        m = {k: float(np.mean(v)) for k, v in agg2[rn].items() if v}
        br8 = float(np.nanmean(agg2[rn]["bitrate_bps_8"])) if agg2[rn].get("bitrate_bps_8") else float("nan")
        L.append(f"| {rn} | {m.get('flops_encode', 0):.2e} | {m.get('flops_decode', 0):.2e} | "
                 f"{m.get('wall_ms_encode', 0):.1f} | {m.get('wall_ms_decode', 0):.1f} | "
                 f"{_fmt_kbps(br8)} |")
    L.append("")

    # ---- Table 3: narrowband sweep (bitrate at 12 bits) ----
    if sweep and "sweep" in sweep:
        bits_label = sweep["sweep"][0].get("bits", 8) if sweep["sweep"] else 8
        L.append(f"## Table 3 — Filter bandwidth sweep ({bits_label}-bit, β=1.0)")
        L.append("")
        L.append("*Varies filter BW at fixed modulation index. Non-monotonic pattern "
                 "reflects filter truncation, NOT Bedrosian validity — see Tables 4-6.*")
        L.append("")
        L.append("| fc (Hz) | BW/fc | env_upper (Hz) | car_lower (Hz) | Bed. ratio | SNR (dB) |")
        L.append("|---|---|---|---|---|---|")
        for row in sweep["sweep"]:
            eu = row.get("env_upper_hz", float("nan"))
            cl = row.get("car_lower_hz", float("nan"))
            bd = row.get("bedrosian_ratio", float("nan"))
            bd_s = "inf" if bd == float("inf") else f"{bd:.2f}"
            L.append(f"| {row['fc_hz']:.0f} | {row['bw_ratio']:.2f} | "
                     f"{eu:.0f} | {cl:.0f} | {bd_s} | {row['snr_db']:.2f} |")
        L.append("")

    # ---- Verdict ----
    L.append("## Verdict")
    L.append("")
    try:
        for b in [8, 12, 16]:
            if ("filterbank_amfm", b) in agg and ("stft", b) in agg:
                a_snr, _ = _ci95(agg[("filterbank_amfm", b)]["snr_db"])
                s_snr, _ = _ci95(agg[("stft", b)]["snr_db"])
                a_br, _ = _ci95(agg[("filterbank_amfm", b)]["bitrate_bps"])
                s_br, _ = _ci95(agg[("stft", b)]["bitrate_bps"])
                L.append(f"- **At {b}-bit**: AM/FM SNR {a_snr:.1f} dB "
                         f"@ {_fmt_kbps(a_br)} kbps vs STFT {s_snr:.1f} dB "
                         f"@ {_fmt_kbps(s_br)} kbps.")
        if "parallel" in agg2:
            p_flops = float(np.mean(agg2["parallel"]["flops_encode"]))
            L.append(f"- Parallel encode: {p_flops:.2e} FLOPs (sum of branches, no discount).")
    except Exception as e:
        L.append(f"(Verdict generation failed: {e})")
    L.append("")

    text = "\n".join(L)
    target = Path("RESULTS.md")
    preserved = ""
    if target.exists():
        existing = target.read_text()
        for marker in ["\n# Extended Analysis", "\n## Table 4",
                       "\n## Consolidated Verdict"]:
            pos = existing.find(marker)
            if pos >= 0:
                preserved = existing[pos:]
                break
    target.write_text(text + preserved)
    return text + preserved
