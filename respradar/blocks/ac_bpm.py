import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Autocorrelation BPM estimator (immune to DC offset and slow drift).

    Every window_sec seconds the mean is removed from the last window_sec of
    signal and its autocorrelation is computed through the FFT
    (Wiener-Khinchin, zero-padded to avoid circular wrap). The first peak in
    the lag range 60/bpm_max .. 60/bpm_min seconds gives the period; a
    parabolic fit on the three lags around it refines it. Peaks with a
    normalised correlation below corr_threshold are ignored. Output is 0.0
    until the first accepted estimate, then holds the last one.
    """

    def __init__(self, samp_rate=500.0, window_sec=60.0,
                 bpm_min=3.0, bpm_max=30.0, corr_threshold=0.3):
        gr.sync_block.__init__(self, name='AC BPM Estimator',
                               in_sig=[np.float32], out_sig=[np.float32])
        self.fs = float(samp_rate)
        self.window_len = int(self.fs * float(window_sec))
        self.bpm_min = float(bpm_min)
        self.bpm_max = float(bpm_max)
        self.corr_threshold = float(corr_threshold)
        self.tau_min = int(self.fs * 60.0 / self.bpm_max)
        self.tau_max = int(self.fs * 60.0 / self.bpm_min)
        self.buffer = np.zeros(self.window_len, dtype=np.float32)
        self.buf_idx = 0
        self.filled = False
        self.current_bpm = 0.0
        self.update_every = self.window_len
        self.since_update = 0

    def _push(self, x):
        n = len(x)
        L = self.window_len
        if n >= L:                      # a chunk bigger than the window replaces it
            self.buffer[:] = x[-L:]
            self.buf_idx = 0
            self.filled = True
            return
        end = self.buf_idx + n
        if end <= L:
            self.buffer[self.buf_idx:end] = x
        else:
            first = L - self.buf_idx
            self.buffer[self.buf_idx:] = x[:first]
            self.buffer[:n - first] = x[first:]
        self.buf_idx = (self.buf_idx + n) % L
        if end >= L:
            self.filled = True

    def _compute_bpm(self):
        sig = np.concatenate((self.buffer[self.buf_idx:], self.buffer[:self.buf_idx]))
        sig = sig - np.mean(sig)
        N = len(sig)
        padded = np.concatenate([sig, np.zeros(N, dtype=np.float32)])
        spec = np.fft.rfft(padded)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:N]
        if ac[0] <= 0:
            return
        ac = ac / ac[0]
        tmin = min(self.tau_min, N - 2)
        tmax = min(self.tau_max, N - 2)
        if tmin >= tmax:
            return
        k = int(np.argmax(ac[tmin:tmax + 1])) + tmin
        if ac[k] < self.corr_threshold:
            return
        if 0 < k < N - 1:
            a, b, c = ac[k - 1], ac[k], ac[k + 1]
            denom = a - 2.0 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            period = (k + delta) / self.fs
        else:
            period = k / self.fs
        if period > 0:
            self.current_bpm = float(np.clip(60.0 / period, 0.0, 60.0))

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        n = len(x)
        self._push(x)
        self.since_update += n
        if self.filled and self.since_update >= self.update_every:
            self._compute_bpm()
            self.since_update = 0
        out[:n] = np.float32(self.current_bpm)
        return n
