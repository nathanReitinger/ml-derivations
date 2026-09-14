"""Builds a tiny model via lightweight fake-torch stubs (no real torch
needed) and exports it with the REAL bridge.export_model_chunked(), so
tests/chunked_e2e.test.js can verify the real Python->JS path without
requiring a torch install in CI. Invoked as a subprocess by that test -
not meant to be run standalone, but works fine that way too.

Usage: python3 dump_chunked_fixture.py <out_dir>
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import numpy as np
from ascii_diffusion import bridge


class FakeT:
    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class FakeLinear:
    def __init__(self, out_f, in_f, rng):
        self.weight = FakeT(rng.normal(0, 0.4, (out_f, in_f)))
        self.bias = FakeT(rng.normal(0, 0.1, (out_f,)))


class FakeConv:
    def __init__(self, out_c, in_c, k, rng):
        self.conv = type("C", (), {})()
        self.conv.weight = FakeT(rng.normal(0, 0.3, (out_c, in_c, k, k)))
        self.conv.bias = FakeT(rng.normal(0, 0.1, (out_c,)))


class FakeNorm:
    def __init__(self, c, rng):
        self.gamma = FakeT(rng.normal(1, 0.05, (c,)))
        self.beta = FakeT(rng.normal(0, 0.05, (c,)))


class FakeFilm:
    def __init__(self, temb_dim, channels, rng):
        self.proj = FakeLinear(channels * 2, temb_dim, rng)


def build_fake_model():
    rng = np.random.default_rng(5)
    H, W, VOCAB, HIDDEN, TIMEDIM = 10, 20, 2, 8, 16

    class M:
        pass

    m = M()
    m.height, m.width, m.vocab_size, m.hidden, m.time_dim = H, W, VOCAB, HIDDEN, TIMEDIM
    m.tDense1 = FakeLinear(128, TIMEDIM, rng)
    m.tDense2 = FakeLinear(HIDDEN, 128, rng)
    m.stemConv = FakeConv(HIDDEN, VOCAB * 2, 3, rng)
    m.stemNorm = FakeNorm(HIDDEN, rng)
    m.stemFilm = FakeFilm(HIDDEN, HIDDEN, rng)
    m.downConv = FakeConv(HIDDEN, HIDDEN, 4, rng)
    m.downNorm = FakeNorm(HIDDEN, rng)
    m.downFilm = FakeFilm(HIDDEN, HIDDEN, rng)

    class Block:
        pass

    blocks = []
    for _ in range(2):
        b = Block()
        b.conv1 = FakeConv(HIDDEN, HIDDEN, 3, rng)
        b.conv1.kernel = 3
        b.norm1 = FakeNorm(HIDDEN, rng)
        b.film1 = FakeFilm(HIDDEN, HIDDEN, rng)
        b.conv2 = FakeConv(HIDDEN, HIDDEN, 3, rng)
        b.norm2 = FakeNorm(HIDDEN, rng)
        b.film2 = FakeFilm(HIDDEN, HIDDEN, rng)
        blocks.append(b)
    m.bottleneck = blocks
    m.refine1Conv = FakeConv(HIDDEN, HIDDEN * 2, 3, rng)
    m.refine1Norm = FakeNorm(HIDDEN, rng)
    m.refine1Film = FakeFilm(HIDDEN, HIDDEN, rng)
    m.refine2Conv = FakeConv(HIDDEN, HIDDEN, 3, rng)
    m.refine2Norm = FakeNorm(HIDDEN, rng)
    m.refine2Film = FakeFilm(HIDDEN, HIDDEN, rng)
    m.convOut = FakeConv(VOCAB, HIDDEN, 3, rng)
    return m


if __name__ == "__main__":
    out_dir = sys.argv[1]
    model = build_fake_model()
    # Small chunk size so even this tiny model splits into several real
    # chunks, exercising the actual multi-fetch path, not a 1-chunk no-op.
    bridge.export_model_chunked(
        model, out_dir, chunk_size_bytes=3000,
        extra_meta={"timesteps": 6, "step": 4242, "meanQuality": 0.91},
    )
