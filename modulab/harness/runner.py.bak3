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
