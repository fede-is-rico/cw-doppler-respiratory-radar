import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Zero-Crossing BPM with peak-to-peak adaptive threshold (auto-scale style)."""

    def __init__(self, samp_rate=500.0, window_sec=15.0,
                 frac_of_pp=0.3, min_pp=0.005):
        gr.sync_block.__init__(self, name='ZC BPM Estimator',
                               in_sig=[np.float32], out_sig=[np.float32])
        self.fs = float(samp_rate)
        self.win_samples = int(self.fs * float(window_sec))
        self.frac = float(frac_of_pp)
        self.min_pp = float(min_pp)
        self.stats_buf = np.zeros(self.win_samples, dtype=np.float32)
        self.stats_idx = 0
        self.stats_full = False
        self.center = 0.0
        self.thresh_high = 0.0
        self.thresh_low = 0.0
        self.have_thresh = False
        self.armed = True
        self.crossings = []
        self.n_abs = 0
        self.bpm = 0.0

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        n = len(x)
        end = self.stats_idx + n
        if end <= self.win_samples:
            self.stats_buf[self.stats_idx:end] = x
        else:
            first = self.win_samples - self.stats_idx
            self.stats_buf[self.stats_idx:] = x[:first]
            self.stats_buf[:n - first] = x[first:]
        self.stats_idx = (self.stats_idx + n) % self.win_samples
        if not self.stats_full and end >= self.win_samples:
            self.stats_full = True
        if self.stats_full:
            hi = float(np.max(self.stats_buf))
            lo = float(np.min(self.stats_buf))
            pp = hi - lo
            if pp >= self.min_pp:
                self.center = (hi + lo) / 2.0
                half = self.frac * pp / 2.0
                self.thresh_high = self.center + half
                self.thresh_low = self.center - half
                self.have_thresh = True
            else:
                self.have_thresh = False
        if self.have_thresh:
            for i in range(n):
                v = x[i]
                if self.armed:
                    if v > self.thresh_high:
                        self.crossings.append(self.n_abs + i)
                        self.armed = False
                else:
                    if v < self.thresh_low:
                        self.armed = True
        self.n_abs += n
        cutoff = self.n_abs - self.win_samples
        self.crossings = [t for t in self.crossings if t >= cutoff]
        if len(self.crossings) >= 2:
            span_sec = (self.crossings[-1] - self.crossings[0]) / self.fs
            if span_sec > 0:
                self.bpm = (len(self.crossings) - 1) * 60.0 / span_sec
        else:
            self.bpm = 0.0
        out[:n] = np.float32(self.bpm)
        return n
