"""Thread-safe ring buffers ("taps") and the GNU Radio sinks that fill them.

Every signal the dashboard shows passes through one of these. GNU Radio work()
threads push chunks; the Qt timer reads the latest N samples at 20 Hz.
"""
import threading

import numpy as np
from gnuradio import gr


class TapBuffer:
    """Fixed-capacity ring buffer with a monotonically increasing write count."""

    def __init__(self, name: str, capacity: int, dtype=np.float32):
        self.name = name
        self.capacity = int(capacity)
        self.dtype = dtype
        self.buf = np.zeros(self.capacity, dtype=dtype)
        self._widx = 0
        self._lock = threading.Lock()

    def push(self, data: np.ndarray):
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:          # keep only the tail of a huge chunk
                data = data[-self.capacity:]
                n = self.capacity
            start = self._widx % self.capacity
            end = start + n
            if end <= self.capacity:
                self.buf[start:end] = data
            else:
                first = self.capacity - start
                self.buf[start:] = data[:first]
                self.buf[:n - first] = data[first:]
            self._widx += n

    def latest(self, n: int) -> np.ndarray:
        """Return a copy of the last *n* samples (fewer if not yet available)."""
        with self._lock:
            total = self._widx
            available = min(n, total, self.capacity)
            if available == 0:
                return np.zeros(0, dtype=self.dtype)
            end = total % self.capacity
            if end == 0:
                return self.buf[self.capacity - available:].copy()
            if available <= end:
                return self.buf[end - available:end].copy()
            tail_start = self.capacity - (available - end)
            return np.concatenate([self.buf[tail_start:], self.buf[:end]])

    @property
    def total_written(self) -> int:
        return self._widx


class probe_sink(gr.sync_block):
    """Sink that pushes every work() chunk into a TapBuffer, optionally decimated."""

    def __init__(self, tap: TapBuffer, decim: int = 1):
        gr.sync_block.__init__(self, name="probe_" + tap.name,
                               in_sig=[tap.dtype], out_sig=[])
        self.tap = tap
        self.decim = max(1, int(decim))

    def work(self, input_items, output_items):
        data = input_items[0]
        self.tap.push(data[::self.decim].copy())
        return len(data)


def make_taps() -> dict:
    """One TapBuffer per dashboard signal. Capacities are in samples at the tap's rate."""
    return {
        # Raw IQ from the SDR (1 MS/s, complex). 500 samples = 0.5 ms = 50 cycles of the IF tone.
        "raw_iq":        TapBuffer("raw_iq", 50_000, np.complex64),
        # Bandpass-filtered breathing phase, 500 S/s. 20 s on screen = 10k samples.
        "breathing":     TapBuffer("breathing", 30_000),
        # Unwrapped phase before the bandpass (for debugging).
        "phase_breath":  TapBuffer("phase_breath", 20_000),
        # |dphi/dn| on the motion path and the detector's 0/1 output, 500 S/s.
        "motion_raw":    TapBuffer("motion_raw", 15_000),
        "motion_detect": TapBuffer("motion_detect", 15_000),
        "phase_motion":  TapBuffer("phase_motion", 20_000),
        # Estimator outputs (piecewise constant, 500 S/s).
        "bpm_fft":       TapBuffer("bpm_fft", 4_000),
        "bpm_zc":        TapBuffer("bpm_zc", 4_000),
        "bpm_ac":        TapBuffer("bpm_ac", 4_000),
    }
