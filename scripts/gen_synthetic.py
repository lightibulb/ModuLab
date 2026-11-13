from pathlib import Path
import numpy as np
from scipy.io import wavfile
from modulab.data.synth import generate_synthetic_corpus


def main():
    out = Path("data/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    corpus = generate_synthetic_corpus(fs=16000, duration_s=3.0)
    for name, sig in corpus.items():
        wavfile.write(str(out / f"{name}.wav"), 16000, (sig * 32767).astype(np.int16))
    print(f"Wrote {len(corpus)} WAVs to {out}")


if __name__ == "__main__":
    main()
