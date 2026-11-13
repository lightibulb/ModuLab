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
