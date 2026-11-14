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
from modulab.metrics.bitrate import bits_per_second
from modulab.metrics.spectral import spectral_upper_edge, spectral_lower_edge
from modulab.data.synth import generate_synthetic_corpus, am_fm_tone


# ---------------------------------------------------------------------
# STFT FM: ridge tracking + phase-vocoder unwrap
# ---------------------------------------------------------------------
def _stft_fm(x, fs, carrier_hz, n_fft=512, hop=128, track_bw_hz=200.0):
    """Ridge-tracking phase-vocoder FM extraction.

    Previous version: fixed bin at carrier_hz — failed above vibrato Nyquist.
    This version: peak bin per frame within a tracking window, plus
    wrapped phase advance per bin (phase vocoder style)."""
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    freqs, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                          boundary="zeros", padded=True)
    mag = np.abs(Z)
    phase = np.angle(Z)
    hop_s = hop / fs

    bin_lo = max(0, int(np.argmin(np.abs(freqs - (carrier_hz - track_bw_hz)))))
    bin_hi = min(len(freqs), int(np.argmin(np.abs(freqs - (carrier_hz + track_bw_hz))) + 1))
    if bin_hi - bin_lo < 1:
        return np.zeros_like(x)

    n_frames = Z.shape[1]
    if n_frames < 2:
        return np.zeros_like(x)

    peaks = np.argmax(mag[bin_lo:bin_hi, :], axis=0) + bin_lo
    peak_phases = phase[peaks, np.arange(n_frames)]
    peak_freqs = freqs[peaks]

    dphase = np.diff(peak_phases)
    dphase = np.angle(np.exp(1j * dphase))  # wrap to (-pi, pi]
    inst_freq = peak_freqs[1:] + dphase / (2 * np.pi * hop_s)

    frame_times = (np.arange(len(inst_freq)) + 1) * hop_s
    audio_times = np.arange(len(x)) / fs
    if len(inst_freq) == 0:
        return np.zeros_like(x)
    fm = np.interp(audio_times, frame_times, inst_freq - carrier_hz,
                   left=inst_freq[0] - carrier_hz,
                   right=inst_freq[-1] - carrier_hz)
    return fm


# ---------------------------------------------------------------------
# Synthetic-signal seeds handled at corpus level
# ---------------------------------------------------------------------
def _run_one(repr_obj, op_obj, x, fs, bits):
    enc, enc_ms, enc_peak = wall_time_ms(lambda: repr_obj.encode(x, fs))
    # FIX: always quantize — the 'bits < 16 else enc' shortcut skipped
    # 16-bit quantization entirely, making 16-bit report float precision.
    enc_q = repr_obj.quantize(enc, bits)
    enc_op = repr_obj.apply_operator(enc_q, op_obj)
    x_out, dec_ms, dec_peak = wall_time_ms(lambda: repr_obj.decode(enc_op))
    n = min(len(x), len(x_out))
    metrics = {}
    for name, fn in METRIC_REG.items():
        try:
            metrics[name] = fn(x[:n], x_out[:n], fs) if name == "mcd_db" else fn(x[:n], x_out[:n])
        except Exception:
            metrics[name] = float("nan")
    try:
        br = bits_per_second(repr_obj, len(x), bits, fs=fs)
    except Exception:
        br = float("nan")
    return {
        "metrics": metrics,
        "wall_ms_encode": enc_ms, "wall_ms_decode": dec_ms,
        "peak_mb_encode": enc_peak, "peak_mb_decode": dec_peak,
        "flops_encode": repr_obj.flops_encode(len(x)),
        "flops_decode": repr_obj.flops_decode(len(x)),
        "bitrate_bps": br,
    }


def run_benchmark(cfg: Config, out_dir: Path):
    fs = cfg.fs
    duration = cfg.duration_s
    seeds = cfg.get("seeds", default=[42])
    repr_names = list((cfg.raw.get("representations") or {}).keys())
    if not repr_names:
        repr_names = ["stft", "filterbank_amfm"]
    ops = list(op_reg.REGISTRY.keys())
    bits_list = cfg.get("quantization", "bit_widths", default=[4, 8, 16])

    results = {"config_hash": cfg.hash, "fs": fs, "duration_s": duration,
               "seeds": seeds, "runs": []}
    n_cells = len(repr_names) * len(ops) * len(bits_list) * len(seeds) * 6
    print(f"  Matrix: {len(repr_names)} repr x {len(ops)} ops x "
          f"{len(bits_list)} bits x {len(seeds)} seeds x 6 signals = {n_cells} runs")

    for seed in seeds:
        signals = generate_synthetic_corpus(fs, duration, seed=seed)
        for rname in repr_names:
            try:
                r = repr_reg.build(rname, cfg)
            except Exception as e:
                print(f"  Skip {rname}: {e}")
                continue
            print(f"  [seed={seed}] [{rname}] {r.info()}")
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
                            rec.update({"repr": rname, "op": oname,
                                        "bits": bits, "signal": sname,
                                        "seed": seed})
                            results["runs"].append(rec)
                        except Exception as e:
                            results["runs"].append({"repr": rname, "op": oname,
                                                    "bits": bits, "signal": sname,
                                                    "seed": seed, "error": str(e)})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"run_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, indent=2, default=float))
    print(f"  Wrote {out_file}")
    return results


# ---------------------------------------------------------------------
# Narrowband sweep: unchanged structure, empirical spectral check added
# ---------------------------------------------------------------------
def run_narrowband_sweep(cfg: Config, out_dir: Path):
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
            fm_depth_ratio = float(mod.get("fm_depth_ratio", 0.25))
            fm_depth = fm_depth_ratio * fc * r
            am_depth = float(mod.get("am_depth", 0.5))
            x = am_fm_tone(fs, duration, fc=fc, am_freq=r * fc / 4,
                           am_depth=am_depth, fm_freq=r * fc / 4,
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
            phase0 = float(res[0])
            res = res - phase0
            fm = np.zeros_like(res)
            fm[1:] = np.diff(res) * fs / (2 * np.pi)

            am_levels = 2 ** bits - 1
            am_max = float(am.max()) + 1e-12
            am_q = np.round(am / am_max * am_levels) / am_levels * am_max
            fm_levels = 2 ** (bits - 1) - 1
            fm_max = float(np.abs(fm).max()) + 1e-12
            fm_q = np.round(fm / fm_max * fm_levels) / fm_levels * fm_max

            t = np.arange(len(x)) / fs
            rec_res = 2 * np.pi * np.cumsum(fm_q) / fs
            rec_phase = phase0 + 2 * np.pi * fc * t + rec_res
            x_rec = am_q * np.cos(rec_phase)

            n_full = min(len(xk), len(x_rec))
            edge = int(0.10 * n_full)
            nn = n_full - 2 * edge
            x_ref = xk[edge:edge + nn]
            x_est = x_rec[edge:edge + nn]
            noise = x_ref - x_est
            snr = 10 * np.log10((np.sum(x_ref ** 2) + 1e-20) /
                                (np.sum(noise ** 2) + 1e-20))
            # Empirical edges of pure envelope and pure carrier
            envelope_pure = 1.0 + am_depth * np.cos(2 * np.pi * (r * fc / 4) * t)
            carrier_pure = np.cos(2 * np.pi * fc * t
                                  + (fm_depth / (r * fc / 4)) *
                                  np.sin(2 * np.pi * (r * fc / 4) * t))
            env_upper_hz = spectral_upper_edge(envelope_pure, fs, thresh_db=-40)
            car_lower_hz = spectral_lower_edge(carrier_pure, fs, thresh_db=-40)
            bed_ratio = (env_upper_hz / car_lower_hz
                         if car_lower_hz > 0 else float("inf"))
            sweep.append({"fc_hz": float(fc), "bw_ratio": float(r),
                          "bw_hz": float(r * fc), "snr_db": float(snr),
                          "env_upper_hz": float(env_upper_hz),
                          "car_lower_hz": float(car_lower_hz),
                          "bedrosian_ratio": float(bed_ratio),
                          "bits": bits})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sweep_{int(time.time())}.json"
    out_file.write_text(json.dumps(sweep, indent=2))
    print(f"  Wrote {out_file}")
    return {"sweep": sweep, "out_file": str(out_file)}


# ---------------------------------------------------------------------
# Bedrosian 1D — empirical spectral support
# ---------------------------------------------------------------------
def run_bedrosian_sweep(cfg: Config, out_dir: Path):
    from scipy.signal import hilbert
    fs = cfg.fs
    duration = cfg.duration_s
    fc = float(cfg.get("bedrosian_sweep", "fc_hz", default=1000.0))
    fm_freq = float(cfg.get("bedrosian_sweep", "fm_freq_hz", default=400.0))
    am_freq = float(cfg.get("bedrosian_sweep", "am_freq_hz", default=400.0))
    am_depth = float(cfg.get("bedrosian_sweep", "am_depth", default=0.5))
    betas = cfg.get("bedrosian_sweep", "betas",
                    default=[0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
    bits = int(cfg.get("bedrosian_sweep", "bits", default=0))

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

        a = hilbert(x)
        am_est = np.abs(a)
        ph_est = np.unwrap(np.angle(a))
        res = ph_est - car_phase
        phase0 = float(res[0])
        res = res - phase0
        fm_est = np.zeros(n, dtype=np.float64)
        fm_est[1:] = np.diff(res) * fs / (2 * np.pi)

        am_true = env
        fm_true = fm_depth * np.cos(2 * np.pi * fm_freq * t)

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

        # EMPIRICAL spectral support of the *pure* envelope and *pure* carrier
        # (not of the composite signal, which spans the whole bandwidth).
        envelope_pure = env  # 1 + am_depth*cos(2π*am_freq*t)
        carrier_pure = np.cos(inst_phase)  # FM carrier without envelope
        env_upper_hz = spectral_upper_edge(envelope_pure, fs, thresh_db=-40)
        car_lower_hz = spectral_lower_edge(carrier_pure, fs, thresh_db=-40)
        bed_ratio = (env_upper_hz / car_lower_hz
                     if car_lower_hz > 0 else float("inf"))

        rows.append({
            "beta": float(beta), "recon_snr_db": float(recon_snr),
            "am_snr_db": float(am_snr_db), "fm_snr_db": float(fm_snr_db),
            "bedrosian_ratio": float(bed_ratio),
            "env_upper_hz": float(env_upper_hz),
            "car_lower_hz": float(car_lower_hz),
            "bits": bits,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bedrosian_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")

    print(f"  {'beta':>6}  {'Bed.':>6}  {'AM SNR':>8}  {'FM SNR':>8}  {'Recon':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in rows:
        bed = r["bedrosian_ratio"]
        bs = f"{bed:>6.2f}" if bed != float("inf") else "   inf"
        print(f"  {r['beta']:>6.2f}  {bs}  {r['am_snr_db']:>7.2f}dB  "
              f"{r['fm_snr_db']:>7.2f}dB  {r['recon_snr_db']:>7.2f}dB")
    return {"bedrosian": rows, "out_file": str(out_file)}


# ---------------------------------------------------------------------
# Bedrosian 2D — empirical spectral support
# ---------------------------------------------------------------------
def run_bedrosian_2d_sweep(cfg: Config, out_dir: Path):
    from scipy.signal import hilbert
    fs = cfg.fs
    duration = cfg.duration_s
    fc = float(cfg.get("bedrosian_2d", "fc_hz", default=1000.0))
    fm_freq = float(cfg.get("bedrosian_2d", "fm_freq_hz", default=400.0))
    am_freq = float(cfg.get("bedrosian_2d", "am_freq_hz", default=400.0))
    betas = cfg.get("bedrosian_2d", "betas",
                    default=[0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    am_depths = cfg.get("bedrosian_2d", "am_depths",
                        default=[0.10, 0.30, 0.50, 0.70, 0.90])

    n = int(fs * duration)
    t = np.arange(n) / fs
    car_phase = 2 * np.pi * fc * t
    edge = int(0.05 * n)

    def _pow_snr(ref, est):
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
        r = np.sqrt(np.mean(ref ** 2)) + 1e-20
        e = np.sqrt(np.mean((est - ref) ** 2)) + 1e-20
        return 20 * np.log10(r / e)

    print(f"  2D sweep: {len(betas)} beta x {len(am_depths)} am_depth")
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

            # EMPIRICAL edges of pure envelope and pure carrier
            envelope_pure = env
            carrier_pure = np.cos(inst_phase)
            env_upper_hz = spectral_upper_edge(envelope_pure, fs, thresh_db=-40)
            car_lower_hz = spectral_lower_edge(carrier_pure, fs, thresh_db=-40)
            bed_ratio = (env_upper_hz / car_lower_hz
                         if car_lower_hz > 0 else float("inf"))

            rows.append({
                "beta": float(beta), "am_depth": float(am_depth),
                "am_snr_db": float(am_snr), "fm_snr_db": float(fm_snr),
                "bedrosian_ratio": float(bed_ratio),
                "env_upper_hz": float(env_upper_hz),
                "car_lower_hz": float(car_lower_hz),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"bedrosian_2d_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")
    return {"bedrosian_2d": rows, "out_file": str(out_file)}


# ---------------------------------------------------------------------
# Speech-like — unchanged (LPF + STFT comparison already correct)
# ---------------------------------------------------------------------
def _speech_like_signal(fs, duration, am_freq, am_depth, f0, formants, seed=42):
    from scipy.signal import hilbert
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs
    env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
    f0_t = f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 0.5 * t))
    phase_fund = 2 * np.pi * np.cumsum(f0_t) / fs
    carrier = np.zeros(n, dtype=np.float64)
    for h in range(1, 60):
        fh = h * f0
        if fh > fs / 2 - 500:
            break
        amp = 0.0
        for fc, bw in formants:
            amp += (bw ** 2) / (bw ** 2 + (fh - fc) ** 2)
        carrier += amp * np.cos(h * phase_fund + rng.uniform(0, 2 * np.pi))
    carrier /= (np.abs(carrier).max() + 1e-12)
    carrier_env = np.abs(hilbert(carrier))
    x = env * carrier
    return x.astype(np.float32), env, carrier_env


def _lpf(sig, fs, cutoff_hz=30.0, order=4):
    from scipy.signal import butter, sosfiltfilt
    nyq = fs / 2.0
    if cutoff_hz >= nyq * 0.95:
        return sig
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, sig)


def _si_sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-20)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-20) /
                         (np.sum(noise ** 2) + 1e-20))


def _stft_envelope(x, fs, n_fft=512, hop=128):
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    _, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                      boundary="zeros", padded=True)
    frame_rms = np.sqrt(np.sum(np.abs(Z) ** 2, axis=0))
    frame_times = np.arange(len(frame_rms)) * hop / fs
    audio_times = np.arange(len(x)) / fs
    return np.interp(audio_times, frame_times, frame_rms,
                     left=float(frame_rms[0]), right=float(frame_rms[-1]))


def _stft_envelope_short(x, fs, n_fft=128, hop=32):
    from scipy.signal import stft as sp_stft
    noverlap = n_fft - hop
    _, _, Z = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=noverlap,
                      boundary="zeros", padded=True)
    frame_rms = np.sqrt(np.sum(np.abs(Z) ** 2, axis=0))
    frame_times = np.arange(len(frame_rms)) * hop / fs
    audio_times = np.arange(len(x)) / fs
    return np.interp(audio_times, frame_times, frame_rms,
                     left=float(frame_rms[0]), right=float(frame_rms[-1]))


def run_speech_like_sweep(cfg: Config, out_dir: Path):
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
    print(f"  Speech-like sweep: {len(am_freqs)} rates")
    rows = []
    for am_freq in am_freqs:
        x, env_true, carrier_env = _speech_like_signal(
            fs, duration, float(am_freq), am_depth, f0, formants, seed=42)
        lpf_eff = min(max(lpf_hz, am_freq * 1.5), 500.0)
        am_hil = np.abs(hilbert(x))
        am_hil_lp = _lpf(am_hil, fs, lpf_eff)
        stft_env = _stft_envelope(x, fs, stft_nfft, stft_hop)
        stft_env_lp = _lpf(stft_env, fs, lpf_eff)
        stft_env_short = _stft_envelope_short(x, fs)
        stft_env_short_lp = _lpf(stft_env_short, fs, lpf_eff)
        am_theory_raw = env_true * carrier_env
        env_true_lp = _lpf(env_true, fs, lpf_eff)
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
            "note": ("lpf-limited" if am_freq > lpf_hz else "in-band"),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"speech_like_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")
    return {"speech_like": rows, "out_file": str(out_file)}


# ---------------------------------------------------------------------
# FM sweep — uses new ridge-tracking _stft_fm (fix #5)
# ---------------------------------------------------------------------
def run_fm_sweep(cfg: Config, out_dir: Path):
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

    print(f"  FM sweep: {len(fm_freqs)} rates, ridge-tracking STFT")
    rows = []
    for fm_freq in fm_freqs:
        env = 1.0 + am_depth * np.cos(2 * np.pi * am_freq * t)
        phase = 2 * np.pi * f0 * t + (fm_depth / fm_freq) * np.sin(2 * np.pi * fm_freq * t)
        x = (env * np.cos(phase)).astype(np.float32)

        a = hilbert(x)
        ph_hil = np.unwrap(np.angle(a))
        fm_hil = np.gradient(ph_hil) * fs / (2 * np.pi) - f0

        # Ridge-tracking at multiple hops
        fm_stft = _stft_fm(x, fs, f0, stft_nfft, stft_hop) - f0
        fm_stft_h32 = _stft_fm(x, fs, f0, stft_nfft, 32) - f0
        fm_stft_h256 = _stft_fm(x, fs, f0, stft_nfft, 256) - f0

        fm_true = fm_depth * np.cos(2 * np.pi * fm_freq * t)
        am_fm_snr = _pow_snr(fm_true[edge:n - edge], fm_hil[edge:n - edge])
        stft_fm_snr = _pow_snr(fm_true[edge:n - edge], fm_stft[edge:n - edge])
        stft_h32_snr = _pow_snr(fm_true[edge:n - edge], fm_stft_h32[edge:n - edge])
        stft_h256_snr = _pow_snr(fm_true[edge:n - edge], fm_stft_h256[edge:n - edge])

        rows.append({
            "fm_freq_hz": float(fm_freq),
            "am_fm_snr_db": float(am_fm_snr),
            "stft_fm_snr_db": float(stft_fm_snr),
            "stft_h32_snr_db": float(stft_h32_snr),
            "stft_h256_snr_db": float(stft_h256_snr),
            "stft_nyq_hz": float(fs / stft_hop / 2.0),
            "above_stft_nyq": bool(fm_freq > fs / stft_hop / 2.0),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"fm_sweep_{int(time.time())}.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"  Wrote {out_file}")
    return {"fm_sweep": rows, "out_file": str(out_file)}
