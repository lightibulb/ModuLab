from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

from modulab.config import Config
from modulab import representations as repr_reg
from modulab import operators as op_reg
from modulab.metrics import REGISTRY as METRIC_REG
from modulab.metrics.cost import wall_time_ms
from modulab.data.synth import generate_synthetic_corpus


def _run_one(repr_obj, op_obj, x, fs, bits):
    enc, enc_ms, enc_peak = wall_time_ms(lambda: repr_obj.encode(x, fs))
    enc_q = repr_obj.quantize(enc, bits) if bits < 16 else enc
    enc_op = repr_obj.apply_operator(enc_q, op_obj)
    x_out, dec_ms, dec_peak = wall_time_ms(lambda: repr_obj.decode(enc_op))
    n = min(len(x), len(x_out))
    metrics = {}
    for name, fn in METRIC_REG.items():
        try:
            metrics[name] = fn(x[:n], x_out[:n], fs) if name == "mcd_db" else fn(x[:n], x_out[:n])
        except Exception:
            metrics[name] = float("nan")
    return {
        "metrics": metrics,
        "wall_ms_encode": enc_ms, "wall_ms_decode": dec_ms,
        "peak_mb_encode": enc_peak, "peak_mb_decode": dec_peak,
        "flops_encode": repr_obj.flops_encode(len(x)),
        "flops_decode": repr_obj.flops_decode(len(x)),
    }


def run_benchmark(cfg: Config, out_dir: Path):
    fs = cfg.fs
    duration = cfg.duration_s
    signals = generate_synthetic_corpus(fs, duration)
    repr_names = list((cfg.raw.get("representations") or {}).keys())
    if not repr_names:
        repr_names = ["stft", "filterbank_amfm"]
    ops = list(op_reg.REGISTRY.keys())
    bits_list = cfg.get("quantization", "bit_widths", default=[4, 8, 16])

    results = {"config_hash": cfg.hash, "fs": fs, "duration_s": duration, "runs": []}
    print(f"  Matrix: {len(repr_names)} repr x {len(ops)} ops x {len(bits_list)} bits x {len(signals)} signals")
    for rname in repr_names:
        try:
            r = repr_reg.build(rname, cfg)
        except Exception as e:
            print(f"  Skip {rname}: {e}")
            continue
        print(f"  [{rname}] {r.info()}")
        for oname in ops:
            try:
                op = op_reg.build(oname)
            except Exception:
                continue
            if not op.compatible(rname) and rname != "parallel":
                continue
            for bits in bits_list:
                for sname, x in signals.items():
                    try:
                        rec = _run_one(r, op, x, fs, bits)
                        rec.update({"repr": rname, "op": oname, "bits": bits, "signal": sname})
                        results["runs"].append(rec)
                    except Exception as e:
                        results["runs"].append({"repr": rname, "op": oname, "bits": bits,
                                                "signal": sname, "error": str(e)})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, indent=2, default=float))
    print(f"  Wrote {out_file}")
    return results


def run_narrowband_sweep(cfg: Config, out_dir: Path):
    from modulab.data.synth import am_fm_tone
    from scipy.signal import butter, sosfiltfilt, hilbert
    fs = cfg.fs
    duration = cfg.duration_s
    fcs = cfg.get("narrowband_sweep", "fc_hz", default=[500, 1000, 2000, 4000])
    ratios = cfg.get("narrowband_sweep", "bw_ratios", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    mod = cfg.get("narrowband_sweep", "modulation", default={})
    bits = int(cfg.get("narrowband_sweep", "bits", default=12))

    sweep = []
    print(f"  Sweep: {len(fcs)} fc x {len(ratios)} bw ratios @ {bits}-bit")
    for fc in fcs:
        for r in ratios:
            # Scale FM depth with r*fc so signal bandwidth tracks filter BW.
            # fm_depth_ratio=0.25 means FM deviation is 25% of target bandwidth.
            fm_depth_ratio = float(mod.get("fm_depth_ratio", 0.25))
            fm_depth = fm_depth_ratio * fc * r
            am_depth = float(mod.get("am_depth", 0.5))
            x = am_fm_tone(fs, duration, fc=fc, am_freq=r * fc / 4,
                           am_depth=am_depth,
                           fm_freq=r * fc / 4,
                           fm_depth=fm_depth)
            nyq = fs / 2.0
            lo = max((fc - r * fc / 2) / nyq, 0.001)
            hi = min((fc + r * fc / 2) / nyq, 0.999)
            sos = butter(4, [lo, hi], btype="band", output="sos")
            xk = sosfiltfilt(sos, x)
            a = hilbert(xk)
            am = np.abs(a)
            ph = np.unwrap(np.angle(a))
            car = 2 * np.pi * fc * np.arange(len(x)) / fs
            res = ph - car
            # CRITICAL: preserve phase0 BEFORE zeroing
            phase0 = float(res[0])
            res = res - phase0
            # First-difference with leading zero: cumsum exactly inverts
            fm = np.zeros_like(res)
            fm[1:] = np.diff(res) * fs / (2 * np.pi)

            # Configurable bit-width quantization
            am_levels = 2 ** bits - 1
            am_max = float(am.max()) + 1e-12
            am_q = np.round(am / am_max * am_levels) / am_levels * am_max
            fm_levels = 2 ** (bits - 1) - 1
            fm_max = float(np.abs(fm).max()) + 1e-12
            fm_q = np.round(fm / fm_max * fm_levels) / fm_levels * fm_max

            # Reconstruct with phase0 restored
            t = np.arange(len(x)) / fs
            rec_res = 2 * np.pi * np.cumsum(fm_q) / fs
            rec_phase = phase0 + 2 * np.pi * fc * t + rec_res
            x_rec = am_q * np.cos(rec_phase)

            # CRITICAL FIX: compare against FILTERED reference (xk), not original (x).
            # Reconstruction is of xk, so xk is the right target.
            # Skip 10% edges to remove sosfiltfilt transient artifacts.
            n_full = min(len(xk), len(x_rec))
            edge = int(0.10 * n_full)
            n = n_full - 2 * edge
            x_ref = xk[edge:edge + n]
            x_est = x_rec[edge:edge + n]
            noise = x_ref - x_est
            snr = 10 * np.log10((np.sum(x_ref ** 2) + 1e-20) /
                                (np.sum(noise ** 2) + 1e-20))
            # Also record full-signal SNR (x vs x_rec) for comparison
            noise_full = x[:n_full] - x_rec[:n_full]
            snr_vs_original = 10 * np.log10(
                (np.sum(x[:n_full] ** 2) + 1e-20) /
                (np.sum(noise_full ** 2) + 1e-20))
            sweep.append({"fc_hz": float(fc), "bw_ratio": float(r),
                          "bw_hz": float(r * fc), "snr_db": float(snr),
                          "snr_vs_original_db": float(snr_vs_original),
                          "bits": bits})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sweep_{int(time.time())}.json"
    out_file.write_text(json.dumps(sweep, indent=2))
    print(f"  Wrote {out_file}")
    if max(s["snr_db"] for s in sweep) < 0:
        print(f"  WARNING: max SNR = {max(s['snr_db'] for s in sweep):.1f} dB")
        print(f"  Expected >10 dB for narrowband cases. Check phase or filter stability.")
    return {"sweep": sweep, "out_file": str(out_file)}


def run_bedrosian_sweep(cfg: Config, out_dir: Path):
    """Bedrosian test: compare hilbert-derived AM/FM to theoretical ground truth.
    Reconstruction SNR is reported but is NOT the metric — it stays flat
    because hilbert(x) always reconstructs x. The real test is decomposition
    error vs. theory."""
    from scipy.signal import hilbert
    fs = cfg.fs
    duration = cfg.duration_s
    fc = float(cfg.get("bedrosian_sweep", "fc_hz", default=1000.0))
    fm_freq = float(cfg.get("bedrosian_sweep", "fm_freq_hz", default=400.0))
    am_freq = float(cfg.get("bedrosian_sweep", "am_freq_hz", default=400.0))
    am_depth = float(cfg.get("bedrosian_sweep", "am_depth", default=0.5))
    betas = cfg.get("bedrosian_sweep", "betas",
                    default=[0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    bits = int(cfg.get("bedrosian_sweep", "bits", default=0))  # 0 => float

    rows = []
    n = int(fs * duration)
    t = np.arange(n) / fs
    car_phase = 2 * np.pi * fc * t
    edge = int(0.05 * n)

    print(f"  Bedrosian sweep: fc={fc} am_freq={am_freq} fm_freq={fm_freq} "
          f"bits={'float' if bits == 0 else bits}")
    for beta in betas:
        fm_depth = beta * fm_freq
        env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
        inst_phase = car_phase + (fm_depth / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
        x = (env * np.cos(inst_phase)).astype(np.float32)

        # hilbert(x) gives analytic AM and phase
        a = hilbert(x)
        am_est = np.abs(a)
        ph_est = np.unwrap(np.angle(a))
        res = ph_est - car_phase
        phase0 = float(res[0])
        res = res - phase0
        fm_est = np.zeros(n, dtype=np.float64)
        fm_est[1:] = np.diff(res) * fs / (2 * np.pi)

        # Theoretical ground truth
        am_true = env
        fm_true = fm_depth * np.cos(2 * np.pi * fm_freq * t)

        # Decomposition error (skip edges — hilbert boundary effects)
        am_true_c = am_true[edge:n-edge]
        am_est_c = am_est[edge:n-edge]
        am_ref_pow = np.sqrt(np.mean(am_true_c ** 2))
        am_err_pow = np.sqrt(np.mean((am_est_c - am_true_c) ** 2)) + 1e-20
        am_snr_db = 20 * np.log10(am_ref_pow / am_err_pow)

        fm_true_c = fm_true[edge:n-edge]
        fm_est_c = fm_est[edge:n-edge]
        fm_ref_pow = np.sqrt(np.mean(fm_true_c ** 2))
        fm_err_pow = np.sqrt(np.mean((fm_est_c - fm_true_c) ** 2)) + 1e-20
        fm_snr_db = 20 * np.log10(fm_ref_pow / fm_err_pow)

        # Reconstruction SNR (for reference — will be flat)
        if bits == 0:
            am_q, fm_q = am_est, fm_est
        else:
            am_lv = 2 ** bits - 1
            am_max = float(am_est.max()) + 1e-12
            am_q = np.round(am_est / am_max * am_lv) / am_lv * am_max
            fm_lv = 2 ** (bits - 1) - 1
            fm_max = float(np.abs(fm_est).max()) + 1e-12
            fm_q = np.round(fm_est / fm_max * fm_lv) / fm_lv * fm_max

        rec_phase = phase0 + car_phase + 2 * np.pi * np.cumsum(fm_q) / fs
        x_rec = am_q * np.cos(rec_phase)
        noise = x - x_rec
        recon_snr = 10 * np.log10((np.sum(x ** 2) + 1e-20) /
                                  (np.sum(noise ** 2) + 1e-20))

        # Bedrosian ratio: envelope_upper / carrier_lower
        env_upper = am_freq * (1 + am_depth)
        carrier_lower = fc - fm_depth
        bed_ratio = env_upper / carrier_lower if carrier_lower > 0 else float("inf")

        rows.append({
            "beta": float(beta),
            "recon_snr_db": float(recon_snr),
            "am_snr_db": float(am_snr_db),
            "fm_snr_db": float(fm_snr_db),
            "bedrosian_ratio": float(bed_ratio),
            "bits": bits,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bedrosian_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    print()
    print(f"  {'beta':>6}  {'Bed.':>6}  {'AM SNR':>8}  {'FM SNR':>8}  {'Recon':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in rows:
        bed = r["bedrosian_ratio"]
        bed_str = f"{bed:>6.2f}" if bed != float("inf") else "   inf"
        print(f"  {r['beta']:>6.2f}  {bed_str}  {r['am_snr_db']:>7.2f}dB  "
              f"{r['fm_snr_db']:>7.2f}dB  {r['recon_snr_db']:>7.2f}dB")

    # Cliff detection on AM/FM decomposition (not recon!)
    am_snrs = [r["am_snr_db"] for r in rows]
    fm_snrs = [r["fm_snr_db"] for r in rows]
    low = [r for r in rows if r["bedrosian_ratio"] < 0.5]
    high = [r for r in rows if r["bedrosian_ratio"] > 1.5]
    print()
    if low and high:
        am_drop = (np.mean([r["am_snr_db"] for r in low]) -
                   np.mean([r["am_snr_db"] for r in high]))
        fm_drop = (np.mean([r["fm_snr_db"] for r in low]) -
                   np.mean([r["fm_snr_db"] for r in high]))
        print(f"  AM SNR drop  (Bed<0.5 -> Bed>1.5): {am_drop:+.1f} dB")
        print(f"  FM SNR drop  (Bed<0.5 -> Bed>1.5): {fm_drop:+.1f} dB")
        print(f"  Recon SNR range                   : "
              f"{max(r['recon_snr_db'] for r in rows) - min(r['recon_snr_db'] for r in rows):.1f} dB")
        if max(am_drop, fm_drop) > 15:
            print("  ✅ Bedrosian cliff visible in decomposition error")
        elif max(am_drop, fm_drop) > 5:
            print("  ⚠️  Weak cliff — but decomposition error IS the right metric")
        else:
            print("  ❌ No cliff even in decomposition error")
    return {"bedrosian": rows, "out_file": str(out_file)}


def run_bedrosian_2d_sweep(cfg: Config, out_dir: Path):
    """2D Bedrosian sweep: beta x am_depth. Measures AM and FM decomposition
    accuracy against ground truth. Expected result: cliff moves to lower beta
    as am_depth increases (envelope spectrum widens)."""
    from scipy.signal import hilbert

    fs = cfg.fs
    duration = cfg.duration_s
    fc = float(cfg.get("bedrosian_2d", "fc_hz", default=1000.0))
    fm_freq = float(cfg.get("bedrosian_2d", "fm_freq_hz", default=400.0))
    am_freq = float(cfg.get("bedrosian_2d", "am_freq_hz", default=400.0))
    betas = cfg.get("bedrosian_2d", "betas",
                    default=[0.05, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00])
    am_depths = cfg.get("bedrosian_2d", "am_depths",
                        default=[0.10, 0.30, 0.50, 0.70, 0.90])

    n = int(fs * duration)
    t = np.arange(n) / fs
    car_phase = 2 * np.pi * fc * t
    edge = int(0.05 * n)

    def _pow_snr(ref, est):
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        ref_pow = np.sqrt(np.mean(ref ** 2)) + 1e-20
        err_pow = np.sqrt(np.mean((est - ref) ** 2)) + 1e-20
        return 20 * np.log10(ref_pow / err_pow)

    print(f"  2D sweep: fc={fc} am_freq={am_freq} fm_freq={fm_freq}")
    print(f"  Grid: {len(betas)} beta x {len(am_depths)} am_depth = "
          f"{len(betas) * len(am_depths)} cells")

    rows = []
    for am_depth in am_depths:
        for beta in betas:
            fm_depth = beta * fm_freq
            env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
            inst_phase = car_phase + (fm_depth / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
            x = (env * np.cos(inst_phase)).astype(np.float32)

            a = hilbert(x)
            am_est = np.abs(a)
            ph_est = np.unwrap(np.angle(a))
            res = ph_est - car_phase
            res = res - res[0]
            fm_est = np.zeros(n, dtype=np.float64)
            fm_est[1:] = np.diff(res) * fs / (2 * np.pi)

            am_true = env
            fm_true = fm_depth * np.cos(2 * np.pi * fm_freq * t)

            am_snr = _pow_snr(am_true[edge:n - edge], am_est[edge:n - edge])
            fm_snr = _pow_snr(fm_true[edge:n - edge], fm_est[edge:n - edge])

            env_upper = am_freq * (1 + am_depth)
            carrier_lower = fc - fm_depth
            bed_ratio = (env_upper / carrier_lower
                         if carrier_lower > 0 else float("inf"))

            rows.append({
                "beta": float(beta),
                "am_depth": float(am_depth),
                "am_snr_db": float(am_snr),
                "fm_snr_db": float(fm_snr),
                "bedrosian_ratio": float(bed_ratio),
                "env_upper_hz": float(env_upper),
                "carrier_lower_hz": float(carrier_lower),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bedrosian_2d_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    # --- ASCII heatmaps ---
    def _snr_char(v):
        if v >= 60: return "█"
        if v >= 40: return "▓"
        if v >= 20: return "▒"
        if v >= 0:  return "░"
        return "·"

    def _print_heatmap(metric_key, title):
        print()
        print(f"  {title}")
        print("         " + "".join(f"{d:>7.2f}" for d in am_depths))
        for b in betas:
            cells = [r for r in rows if r["beta"] == b]
            cells = sorted(cells, key=lambda z: z["am_depth"])
            line = f"  {b:>6.2f}  "
            for c in cells:
                line += f"   {_snr_char(c[metric_key])}   "
            print(line)

    _print_heatmap("am_snr_db", "AM SNR (rows=beta, cols=am_depth)")
    _print_heatmap("fm_snr_db", "FM SNR (rows=beta, cols=am_depth)")

    # --- Numeric table (compact) ---
    print()
    print(f"  AM SNR table (dB)")
    print(f"  {'beta':>6}  " + "  ".join(f"{d:>6.2f}" for d in am_depths))
    print(f"  {'-'*6}  " + "  ".join(f"{'-'*6}" for _ in am_depths))
    for b in betas:
        cells = sorted([r for r in rows if r["beta"] == b],
                       key=lambda z: z["am_depth"])
        print(f"  {b:>6.2f}  " +
              "  ".join(f"{c['am_snr_db']:>6.1f}" for c in cells))

    # --- Cliff analysis: where does AM SNR drop below 20 dB? ---
    print()
    print("  Cliff boundary (first beta where AM SNR < 20 dB per am_depth):")
    for d in am_depths:
        cells = sorted([r for r in rows if r["am_depth"] == d],
                       key=lambda z: z["beta"])
        cliff_beta = None
        for c in cells:
            if c["am_snr_db"] < 20:
                cliff_beta = c["beta"]
                break
        label = f"beta={cliff_beta:.2f}" if cliff_beta is not None else "no cliff"
        print(f"    am_depth={d:.2f}: {label}")

    # --- Matplotlib PNG if available ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, key, title in [
            (axes[0], "am_snr_db", "AM decomposition SNR (dB)"),
            (axes[1], "fm_snr_db", "FM decomposition SNR (dB)"),
        ]:
            grid = np.zeros((len(betas), len(am_depths)))
            for i, b in enumerate(betas):
                for j, d in enumerate(am_depths):
                    cell = next(r for r in rows
                                if r["beta"] == b and r["am_depth"] == d)
                    grid[i, j] = cell[key]
            im = ax.imshow(grid, aspect="auto", origin="lower",
                           cmap="viridis",
                           extent=[min(am_depths), max(am_depths),
                                   min(betas), max(betas)])
            ax.set_xlabel("am_depth")
            ax.set_ylabel("beta (FM modulation index)")
            ax.set_title(title)
            fig.colorbar(im, ax=ax)

        fig.tight_layout()
        png = out_dir / "bedrosian_2d.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {png}")
    except Exception as e:
        print(f"\n  (matplotlib unavailable: {e})")

    return {"bedrosian_2d": rows, "out_file": str(out_file)}


def _speech_like_signal(fs, duration, am_freq, am_depth, f0, formants, seed=42):
    """Synthetic speech-like signal: pitch harmonics shaped by formants,
    modulated by a syllabic envelope. Returns (x, env_true, carrier_env)."""
    from scipy.signal import hilbert
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs

    # Syllabic envelope
    env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)

    # Slow pitch contour (2% vibrato like natural speech)
    f0_t = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t))
    phase_fund = 2 * np.pi * np.cumsum(f0_t) / fs

    # Harmonic carrier with formant-shaped amplitude
    carrier = np.zeros(n, dtype=np.float64)
    for h in range(1, 60):
        fh = h * f0
        if fh > fs / 2 - 500:
            break
        # Lorentzian sum over formants
        amp = 0.0
        for fc, bw in formants:
            amp += (bw ** 2) / (bw ** 2 + (fh - fc) ** 2)
        carrier += amp * np.cos(h * phase_fund + rng.uniform(0, 2 * np.pi))

    carrier /= (np.abs(carrier).max() + 1e-12)

    # Theoretical composite envelope: |hilbert(carrier)| (Bedrosian element-wise)
    carrier_env = np.abs(hilbert(carrier))

    x = env * carrier
    return x.astype(np.float32), env, carrier_env


def run_speech_like_sweep(cfg: Config, out_dir: Path):
    """Sweep syllabic envelope rate on a multi-formant speech-like signal.
    Measures how well the Hilbert envelope recovers (a) theoretical composite
    envelope, (b) pure syllabic envelope."""
    from scipy.signal import hilbert

    fs = cfg.fs
    duration = cfg.duration_s
    f0 = float(cfg.get("speech_like", "f0_hz", default=120.0))
    am_depth = float(cfg.get("speech_like", "am_depth", default=0.5))
    formants = cfg.get("speech_like", "formants",
                       default=[[700, 100], [1220, 120], [2600, 150]])
    formants = tuple((float(f), float(b)) for f, b in formants)
    am_freqs = cfg.get("speech_like", "am_freq_hz",
                       default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 700])

    n = int(duration * fs)
    edge = int(0.10 * n)

    print(f"  Speech-like: f0={f0} Hz, am_depth={am_depth}, formants={formants}")
    print(f"  Sweep: {len(am_freqs)} am_freq values")

    rows = []
    for am_freq in am_freqs:
        x, env_true, carrier_env = _speech_like_signal(
            fs, duration, float(am_freq), am_depth, f0, formants, seed=42
        )
        am_est = np.abs(hilbert(x))

        # Theoretical composite envelope
        am_theory = env_true * carrier_env

        # SNR vs theory (isolates Bedrosian failure)
        ref_c = am_theory[edge:n - edge]
        est_c = am_est[edge:n - edge]
        ref_pow = np.sqrt(np.mean(ref_c ** 2)) + 1e-20
        err_pow = np.sqrt(np.mean((est_c - ref_c) ** 2)) + 1e-20
        snr_vs_theory = 20 * np.log10(ref_pow / err_pow)

        # SNR vs pure syllabic envelope (speech-realistic)
        env_c = env_true[edge:n - edge]
        env_pow = np.sqrt(np.mean(env_c ** 2)) + 1e-20
        err_env = np.sqrt(np.mean((est_c - env_c) ** 2)) + 1e-20
        snr_vs_env = 20 * np.log10(env_pow / err_env)

        # Bandwidth overlap
        env_upper = am_freq * (1 + am_depth)
        f1_lower = formants[0][0] - formants[0][1] / 2
        overlap = env_upper / f1_lower

        rows.append({
            "am_freq_hz": float(am_freq),
            "am_vs_theory_db": float(snr_vs_theory),
            "am_vs_env_db": float(snr_vs_env),
            "envelope_upper_hz": float(env_upper),
            "f1_lower_hz": float(f1_lower),
            "overlap_ratio": float(overlap),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"speech_like_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    # Table
    print()
    print(f"  {'am_freq':>8}  {'env_up':>7}  {'F1_low':>7}  {'ratio':>6}  "
          f"{'AM_thry':>9}  {'AM_env':>8}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*8}")
    for r in rows:
        print(f"  {r['am_freq_hz']:>8.1f}  {r['envelope_upper_hz']:>7.0f}  "
              f"{r['f1_lower_hz']:>7.0f}  {r['overlap_ratio']:>6.2f}  "
              f"{r['am_vs_theory_db']:>8.1f}dB  {r['am_vs_env_db']:>7.1f}dB")

    # Cliff analysis
    low_rate = [r for r in rows if r["am_freq_hz"] <= 8]
    high_rate = [r for r in rows if r["am_freq_hz"] >= 128]
    if low_rate and high_rate:
        drop_theory = (np.mean([r["am_vs_theory_db"] for r in low_rate]) -
                       np.mean([r["am_vs_theory_db"] for r in high_rate]))
        print()
        print(f"  Speech-rate (<=8 Hz) AM SNR vs theory: "
              f"{np.mean([r['am_vs_theory_db'] for r in low_rate]):+.1f} dB")
        print(f"  Consonant-rate (>=128 Hz) AM SNR vs theory: "
              f"{np.mean([r['am_vs_theory_db'] for r in high_rate]):+.1f} dB")
        print(f"  Drop: {drop_theory:+.1f} dB")
        if drop_theory > 20:
            print("  ✅ Speech-like Bedrosian cliff confirmed")
        elif drop_theory > 10:
            print("  ⚠️  Moderate cliff")
        else:
            print("  ⚠️  Weak cliff")

    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        freqs = [r["am_freq_hz"] for r in rows]
        ax.semilogx(freqs, [r["am_vs_theory_db"] for r in rows],
                    "o-", label="AM SNR vs theory")
        ax.semilogx(freqs, [r["am_vs_env_db"] for r in rows],
                    "s-", label="AM SNR vs syllabic env")
        # Mark F1 lower skirt
        f1_lower = formants[0][0] - formants[0][1] / 2
        ax.axvline(f1_lower, color="red", linestyle="--", alpha=0.6,
                   label=f"F1 lower skirt ({f1_lower:.0f} Hz)")
        ax.axvspan(2, 8, alpha=0.1, color="green", label="syllable rate")
        ax.axvspan(50, 200, alpha=0.1, color="orange", label="consonant rate")
        ax.set_xlabel("Syllabic envelope frequency (Hz, log scale)")
        ax.set_ylabel("AM estimation SNR (dB)")
        ax.set_title("Speech-like signal: AM decomposition vs envelope rate")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png = out_dir / "speech_like_sweep.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {png}")
    except Exception as e:
        print(f"\n  (matplotlib unavailable: {e})")

    return {"speech_like": rows, "out_file": str(out_file)}
