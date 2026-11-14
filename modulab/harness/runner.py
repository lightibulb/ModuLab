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
    """Speech-like AM sweep + STFT envelope comparison.
    AM_env is now LPF'd on both sides for fair comparison at syllable rates."""
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
    lpf_hz = float(cfg.get("speech_like", "lpf_hz", default=30.0))
    stft_nfft = int(cfg.get("speech_like", "stft_n_fft", default=512))
    stft_hop = int(cfg.get("speech_like", "stft_hop", default=128))

    n = int(duration * fs)
    edge = int(0.10 * n)

    def _pow_snr(ref, est):
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        r = np.sqrt(np.mean(ref ** 2)) + 1e-20
        e = np.sqrt(np.mean((est - ref) ** 2)) + 1e-20
        return 20 * np.log10(r / e)

    print(f"  Speech-like AM sweep: f0={f0} am_depth={am_depth} "
          f"LPF={lpf_hz} Hz, STFT n_fft={stft_nfft} hop={stft_hop}")

    rows = []
    for am_freq in am_freqs:
        x, env_true, carrier_env = _speech_like_signal(
            fs, duration, float(am_freq), am_depth, f0, formants, seed=42
        )

        # ADAPTIVE LPF: at least am_freq * 1.5, but no more than 500 Hz.
        # This allows high-rate modulations to be tracked.
        lpf_eff = min(max(lpf_hz, am_freq * 1.5), 500.0)

        # AM/FM: LPF(|hilbert(x)|)
        am_hil = np.abs(hilbert(x))
        am_hil_lp = _lpf(am_hil, fs, lpf_eff)

        # STFT (long window): 32 ms window -> ~16 Hz envelope bandwidth
        stft_env = _stft_envelope(x, fs, stft_nfft, stft_hop)
        stft_env_lp = _lpf(stft_env, fs, lpf_eff)

        # STFT (short window): 8 ms window -> ~62 Hz envelope bandwidth
        stft_env_short = _stft_envelope_short(x, fs)
        stft_env_short_lp = _lpf(stft_env_short, fs, lpf_eff)

        # Theoretical envelope (raw composite)
        am_theory_raw = env_true * carrier_env

        # LPF'd true syllabic envelope (fair comparison at speech rates)
        env_true_lp = _lpf(env_true, fs, lpf_eff)

        # Metrics: use SI-SDR (scale-invariant) to normalize the carrier
        # envelope mean that scaled |hilbert(x)| and STFT frame RMS
        am_vs_theory = _si_sdr(am_theory_raw[edge:n - edge], am_hil[edge:n - edge])
        am_env_lp = _si_sdr(env_true_lp[edge:n - edge], am_hil_lp[edge:n - edge])
        stft_env_lp_snr = _si_sdr(env_true_lp[edge:n - edge], stft_env_lp[edge:n - edge])
        stft_env_short_snr = _si_sdr(env_true_lp[edge:n - edge], stft_env_short_lp[edge:n - edge])
        stft_env_raw = _si_sdr(am_theory_raw[edge:n - edge], stft_env[edge:n - edge])

        rows.append({
            "am_freq_hz": float(am_freq),
            "am_vs_theory_db": float(am_vs_theory),
            "am_env_lp_db": float(am_env_lp),
            "stft_env_lp_db": float(stft_env_lp_snr),
            "stft_env_short_db": float(stft_env_short_snr),
            "lpf_eff_hz": float(lpf_eff),
            "stft_vs_theory_db": float(stft_env_raw),
            "note": (
                "lpf-limited" if am_freq > lpf_hz else "in-band"
            ),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"speech_like_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    print()
    print(f"  {'am_freq':>8}  {'LPF':>5}  {'AM_Thy':>7}  {'AM_env':>7}  "
          f"{'STFT_lng':>8}  {'STFT_sht':>8}")
    print(f"  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}")
    for r in rows:
        print(f"  {r['am_freq_hz']:>8.1f}  {r['lpf_eff_hz']:>5.0f}  "
              f"{r['am_vs_theory_db']:>6.1f}dB  {r['am_env_lp_db']:>6.1f}dB  "
              f"{r['stft_env_lp_db']:>7.1f}dB  {r['stft_env_short_db']:>7.1f}dB")

    # Speech-rate comparison
    speech = [r for r in rows if r["am_freq_hz"] <= 16]
    if speech:
        am_mean = np.mean([r['am_env_lp_db'] for r in speech])
        stft_mean = np.mean([r['stft_env_lp_db'] for r in speech])
        stft_short = np.mean([r['stft_env_short_db'] for r in speech])
        print()
        print(f"  Speech-rate (am_freq <= 16 Hz) mean, SI-SDR:")
        print(f"    AM_env:        {am_mean:+6.1f} dB")
        print(f"    STFT (long):   {stft_mean:+6.1f} dB")
        print(f"    STFT (short):  {stft_short:+6.1f} dB")
        best_stft = max(stft_mean, stft_short)
        if am_mean > best_stft + 2:
            print(f"    → AM/FM beats best STFT by {am_mean - best_stft:.1f} dB")
        elif best_stft > am_mean + 2:
            print(f"    → STFT beats AM/FM by {best_stft - am_mean:.1f} dB")
        else:
            print(f"    → Tie (within 2 dB) — envelope estimation is representation-agnostic")

    # Note the lpf-limited regime
    above_lpf = [r for r in rows if r["am_freq_hz"] > lpf_hz]
    if above_lpf:
        print()
        print(f"  Above LPF cutoff ({lpf_hz:.0f} Hz): "
              f"{len(above_lpf)} points filtered out by design")
        print("  (raises LPF cutoff in config to test higher rates)")

    # PNG if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        freqs = [r["am_freq_hz"] for r in rows]
        ax.semilogx(freqs, [r["am_vs_theory_db"] for r in rows],
                    "o-", label="AM vs theory")
        ax.semilogx(freqs, [r["am_env_lp_db"] for r in rows],
                    "s-", label=f"AM vs env (LPF {lpf_hz:.0f} Hz)")
        ax.semilogx(freqs, [r["stft_env_lp_db"] for r in rows],
                    "^-", label=f"STFT vs env (LPF {lpf_hz:.0f} Hz)")
        ax.axvline(f0, color="red", linestyle="--", alpha=0.5,
                   label=f"f0 ({f0:.0f} Hz)")
        ax.axvline(lpf_hz, color="gray", linestyle=":", alpha=0.5,
                   label=f"LPF cutoff ({lpf_hz:.0f} Hz)")
        ax.set_xlabel("Syllabic envelope frequency (Hz)")
        ax.set_ylabel("AM SNR (dB)")
        ax.set_title("Speech-like: AM/FM vs STFT envelope recovery")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png = out_dir / "speech_like_sweep.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {png}")
    except Exception as e:
        print(f"\n  (matplotlib unavailable: {e})")

    return {"speech_like": rows, "out_file": str(out_file)}



def _lpf(sig, fs, cutoff_hz=30.0, order=4):
    """Low-pass filter for envelope follower."""
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    if cutoff_hz >= nyq * 0.95:
        return sig
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, sig)


def _si_sdr(ref, est):
    """Scale-invariant SDR in dB (like the standard metric)."""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-20) /
                         (np.sum(noise ** 2) + 1e-20))


def _stft_envelope(x, fs, n_fft=512, hop=128):
    """STFT frame RMS, interpolated back to audio rate."""
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    _, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                      boundary="zeros", padded=True)
    frame_rms = np.sqrt(np.sum(np.abs(Z) ** 2, axis=0))
    frame_times = np.arange(len(frame_rms)) * hop / fs
    audio_times = np.arange(len(x)) / fs
    env = np.interp(audio_times, frame_times, frame_rms,
                    left=float(frame_rms[0]), right=float(frame_rms[-1]))
    return env


def _stft_envelope_short(x, fs, n_fft=128, hop=32):
    """Short-window STFT frame RMS: 8 ms window -> ~62 Hz envelope bandwidth.
    Fair comparison against AM/FM with LPF up to 60 Hz."""
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    _, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                      boundary="zeros", padded=True)
    frame_rms = np.sqrt(np.sum(np.abs(Z) ** 2, axis=0))
    frame_times = np.arange(len(frame_rms)) * hop / fs
    audio_times = np.arange(len(x)) / fs
    return np.interp(audio_times, frame_times, frame_rms,
                     left=float(frame_rms[0]), right=float(frame_rms[-1]))


def _stft_fm(x, fs, carrier_hz, n_fft=512, hop=128):
    """Extract FM from STFT phase at peak bin near carrier_hz."""
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    freqs, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                          boundary="zeros", padded=True)
    bin_idx = int(np.argmin(np.abs(freqs - carrier_hz)))
    phase = np.unwrap(np.angle(Z[bin_idx, :]))
    hop_s = hop / fs
    fm_frames = np.gradient(phase) / (2 * np.pi * hop_s)
    frame_times = np.arange(len(fm_frames)) * hop_s
    audio_times = np.arange(len(x)) / fs
    fm = np.interp(audio_times, frame_times, fm_frames,
                   left=float(fm_frames[0]), right=float(fm_frames[-1]))
    return fm


def run_fm_sweep(cfg: Config, out_dir: Path):
    """FM sweep: sweep pitch vibrato rate, compare AM/FM phase-derivative FM
    to theoretical FM. Compare to STFT phase-derivative FM."""
    from scipy.signal import hilbert

    fs = cfg.fs
    duration = cfg.duration_s
    f0 = float(cfg.get("fm_sweep", "f0_hz", default=1000.0))
    fm_depth = float(cfg.get("fm_sweep", "fm_depth_hz", default=50.0))
    am_freq = float(cfg.get("fm_sweep", "am_freq_hz", default=2.0))
    am_depth = float(cfg.get("fm_sweep", "am_depth", default=0.2))
    fm_freqs = cfg.get("fm_sweep", "fm_freq_hz",
                       default=[0.5, 1, 2, 4, 8, 16, 32, 64, 128, 200])
    stft_nfft = int(cfg.get("fm_sweep", "stft_n_fft", default=512))
    stft_hop = int(cfg.get("fm_sweep", "stft_hop", default=128))

    n = int(duration * fs)
    t = np.arange(n) / fs
    edge = int(0.10 * n)

    def _pow_snr(ref, est):
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        r = np.sqrt(np.mean(ref ** 2)) + 1e-20
        e = np.sqrt(np.mean((est - ref) ** 2)) + 1e-20
        return 20 * np.log10(r / e)

    print(f"  FM sweep: f0={f0} Hz, fm_depth={fm_depth} Hz, "
          f"STFT n_fft={stft_nfft} hop={stft_hop}")

    rows = []
    for fm_freq in fm_freqs:
        # Signal: env * cos(2π f0 t + (fm_depth/fm_freq) sin(2π fm_freq t))
        env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
        phase = 2 * np.pi * f0 * t + (fm_depth / fm_freq) * np.sin(
            2 * np.pi * fm_freq * t)
        x = (env * np.cos(phase)).astype(np.float32)

        # AM/FM extraction
        a = hilbert(x)
        ph_hil = np.unwrap(np.angle(a))
        fm_hil = np.gradient(ph_hil) * fs / (2 * np.pi) - f0

        # STFT extraction at multiple hops
        fm_stft = _stft_fm(x, fs, f0, stft_nfft, stft_hop) - f0
        fm_stft_h32 = _stft_fm(x, fs, f0, stft_nfft, 32) - f0
        fm_stft_h256 = _stft_fm(x, fs, f0, stft_nfft, 256) - f0

        # Theoretical FM (deviation from f0)
        fm_true = fm_depth * np.cos(2 * np.pi * fm_freq * t)

        # Metrics
        am_fm_snr = _pow_snr(fm_true[edge:n - edge], fm_hil[edge:n - edge])
        stft_fm_snr = _pow_snr(fm_true[edge:n - edge], fm_stft[edge:n - edge])
        stft_h32_snr = _pow_snr(fm_true[edge:n - edge], fm_stft_h32[edge:n - edge])
        stft_h256_snr = _pow_snr(fm_true[edge:n - edge], fm_stft_h256[edge:n - edge])

        # STFT frame rate limit
        stft_frame_rate = fs / stft_hop
        stft_nyq = stft_frame_rate / 2.0

        rows.append({
            "fm_freq_hz": float(fm_freq),
            "am_fm_snr_db": float(am_fm_snr),
            "stft_fm_snr_db": float(stft_fm_snr),
            "stft_h32_snr_db": float(stft_h32_snr),
            "stft_h256_snr_db": float(stft_h256_snr),
            "stft_nyq_hz": float(stft_nyq),
            "above_stft_nyq": bool(fm_freq > stft_nyq),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fm_sweep_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    print()
    print(f"  {'fm_freq':>8}  {'AM_FM':>7}  {'STFT_h32':>9}  {'STFT_h128':>10}  "
          f"{'STFT_h256':>10}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*10}")
    for r in rows:
        print(f"  {r['fm_freq_hz']:>8.1f}  {r['am_fm_snr_db']:>6.1f}dB  "
              f"{r['stft_h32_snr_db']:>8.1f}dB  "
              f"{r['stft_fm_snr_db']:>9.1f}dB  "
              f"{r['stft_h256_snr_db']:>9.1f}dB")

    # Summary
    low = [r for r in rows if r["fm_freq_hz"] <= 16]
    high = [r for r in rows if r["fm_freq_hz"] >= 64]
    if low and high:
        print()
        print(f"  Vibrato-rate (<=16 Hz):")
        print(f"    AM/FM FM SNR: {np.mean([r['am_fm_snr_db'] for r in low]):+6.1f} dB")
        print(f"    STFT FM SNR : {np.mean([r['stft_fm_snr_db'] for r in low]):+6.1f} dB")
        print(f"  Fast-rate (>=64 Hz):")
        print(f"    AM/FM FM SNR: {np.mean([r['am_fm_snr_db'] for r in high]):+6.1f} dB")
        print(f"    STFT FM SNR : {np.mean([r['stft_fm_snr_db'] for r in high]):+6.1f} dB")

    # PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        freqs = [r["fm_freq_hz"] for r in rows]
        ax.semilogx(freqs, [r["am_fm_snr_db"] for r in rows],
                    "o-", label="AM/FM phase derivative")
        ax.semilogx(freqs, [r["stft_fm_snr_db"] for r in rows],
                    "s-", label="STFT phase derivative")
        stft_nyq = rows[0]["stft_nyq_hz"]
        ax.axvline(stft_nyq, color="red", linestyle="--", alpha=0.5,
                   label=f"STFT Nyquist ({stft_nyq:.0f} Hz)")
        ax.set_xlabel("FM rate (Hz)")
        ax.set_ylabel("FM estimation SNR (dB)")
        ax.set_title("FM extraction: AM/FM vs STFT")
        ax.grid(True, alpha=0.3)
        ax.legend()
        png = out_dir / "fm_sweep.png"
        fig.savefig(png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Wrote {png}")
    except Exception as e:
        print(f"\n  (matplotlib unavailable: {e})")

    return {"fm_sweep": rows, "out_file": str(out_file)}
