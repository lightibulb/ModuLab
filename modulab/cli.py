"""ModuLab CLI — argparse-based, zero external dependencies."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from modulab.config import Config
from modulab.harness import run_benchmark, run_narrowband_sweep
from modulab.harness.report import generate_report
from modulab import representations as repr_reg
from modulab import operators as op_reg


def cmd_bench(args):
    cfg = Config.load(args.config)
    print(f"Config hash: {cfg.hash}")
    results = run_benchmark(cfg, Path("results/runs"))
    sweep = run_narrowband_sweep(cfg, Path("results"))
    generate_report(results, sweep, Path("results/figures"), Path(args.config))
    print("Done. See RESULTS.md")


def cmd_sweep(args):
    cfg = Config.load(args.config)
    run_narrowband_sweep(cfg, Path("results"))


def cmd_representations(args):
    for name in repr_reg.REGISTRY:
        print(name)


def cmd_operators(args):
    for name in op_reg.REGISTRY:
        print(name)



def cmd_bedrosian2d(args):
    cfg = Config.load(args.config)
    run_bedrosian_2d_sweep(cfg, Path("results"))


def cmd_speech_like(args):
    cfg = Config.load(args.config)
    run_speech_like_sweep(cfg, Path("results"))

def main(argv=None):
    p = argparse.ArgumentParser(prog="modulab", description="AM/FM vs STFT benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench")
    b.add_argument("--config", default="modulab_config.yaml")
    b.set_defaults(fn=cmd_bench)

    s = sub.add_parser("sweep")
    s.add_argument("--config", default="modulab_config.yaml")
    s.set_defaults(fn=cmd_sweep)

    sub.add_parser("representations").set_defaults(fn=cmd_representations)
    sub.add_parser("operators").set_defaults(fn=cmd_operators)

    args = p.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
