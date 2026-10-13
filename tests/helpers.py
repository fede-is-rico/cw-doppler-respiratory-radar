"""Shared helpers for driving blocks without a running flowgraph."""
import numpy as np

FS = 500.0


def run_block(blk, x, chunk=4096):
    """Feed x through blk.work() in chunks and return the concatenated output."""
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    for i in range(0, len(x), chunk):
        xin = x[i:i + chunk]
        o = np.zeros(len(xin), dtype=np.float32)
        n = blk.work([xin], [o])
        assert n == len(xin)
        out[i:i + chunk] = o
    return out


def breathing(bpm, seconds, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(int(seconds * fs)) / fs
    return (amp * np.sin(2 * np.pi * bpm / 60.0 * t + phase)).astype(np.float32)


def bandpassed_noise(seconds, fs=FS, std=1.0, seed=0):
    """White noise through the same 0.1-0.5 Hz FIR the flowgraph uses."""
    from scipy.signal import firwin, lfilter
    from respradar import params as P
    rng = np.random.default_rng(seed)
    taps = firwin(P.BP_TAPS, [P.BP_LOW_HZ, P.BP_HIGH_HZ], pass_zero=False, fs=fs)
    n = int(seconds * fs)
    return lfilter(taps, 1.0, rng.standard_normal(n + P.BP_TAPS) * std)[P.BP_TAPS:].astype(np.float32)
