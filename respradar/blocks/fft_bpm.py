import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """FFT BPM estimator (Blackman-Harris window + linear detrend)."""

    def __init__(self, samp_rate=500.0, window_sec=20.0,
                 fft_size=8192, f_low=0.1, f_high=0.5):
        gr.sync_block.__init__(self, name='FFT BPM Estimator',
                               in_sig=[np.float32], out_sig=[np.float32])
        self.samp_rate = float(samp_rate)
        self.window_len = int(self.samp_rate * float(window_sec))
        self.fft_size = int(fft_size)
        self.buffer = np.zeros(self.window_len, dtype=np.float32)
        self.buf_idx = 0
        self.filled = False
        self.current_bpm = 0.0
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0/self.samp_rate)
        self.band_mask = (self.freqs >= float(f_low)) & (self.freqs <= float(f_high))
        N = self.window_len
        n_arr = np.arange(N, dtype=np.float64)
        self.win = (0.35875
                    - 0.48829 * np.cos(2.0*np.pi*n_arr/(N-1))
                    + 0.14128 * np.cos(4.0*np.pi*n_arr/(N-1))
                    - 0.01168 * np.cos(6.0*np.pi*n_arr/(N-1))).astype(np.float32)
        self.bin_hz = self.freqs[1] - self.freqs[0]
        self.detrend_t = np.arange(self.window_len, dtype=np.float32)
        self.update_every = int(self.samp_rate * float(window_sec))
        self.since_update = 0

    def _compute_bpm(self):
        sig = np.concatenate((self.buffer[self.buf_idx:], self.buffer[:self.buf_idx]))
        coef = np.polyfit(self.detrend_t, sig, 1)
        sig = sig - (coef[0] * self.detrend_t + coef[1])
        spec = np.abs(np.fft.rfft(sig * self.win, n=self.fft_size))**2
        band = np.where(self.band_mask, spec, 0.0)
        k = int(np.argmax(band))
        if k == 0 or k >= len(spec) - 1:
            return
        band_vals = spec[self.band_mask]
        med = np.median(band_vals) + 1e-30
        if band[k] < 6.0 * med:
            return
        a = np.log(spec[k-1] + 1e-30)
        b = np.log(spec[k] + 1e-30)
        c = np.log(spec[k+1] + 1e-30)
        d = a - 2.0*b + c
        delta = 0.5*(a - c)/d if d != 0 else 0.0
        peak_freq = self.freqs[k] + delta * self.bin_hz
        self.current_bpm = float(np.clip(peak_freq * 60.0, 0.0, 60.0))

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        n = len(x)
        end = self.buf_idx + n
        if end <= self.window_len:
            self.buffer[self.buf_idx:end] = x
        else:
            first = self.window_len - self.buf_idx
            self.buffer[self.buf_idx:] = x[:first]
            self.buffer[:n - first] = x[first:]
        self.buf_idx = (self.buf_idx + n) % self.window_len
        if not self.filled and end >= self.window_len:
            self.filled = True
        self.since_update += n
        if self.filled and self.since_update >= self.update_every:
            self._compute_bpm()
            self.since_update = 0
        out[:n] = np.float32(self.current_bpm)
        return n
