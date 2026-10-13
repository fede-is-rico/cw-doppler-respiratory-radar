import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """FFT BPM estimator (Hann window, linear detrend, log-parabolic peak interpolation).

    Every window_sec seconds the last window_sec of the bandpassed breathing
    phase is detrended, Hann-windowed, zero-padded to fft_size and transformed.
    The strongest bin in [f_low, f_high] is refined with a parabolic fit on the
    log spectrum and reported in BPM. An estimate is accepted only when

      * the record's peak-to-peak amplitude is at least min_pp radians, and
      * the peak is at least snr_gate times the median of the in-band bins
        that lie outside the window's main lobe (+/- 2/T Hz for Hann).

    The second rule is what makes such a narrow band workable. A 15 s record
    has a main lobe about 0.27 Hz wide, most of the 0.1-0.5 Hz band, so
    comparing the peak with the median of *all* in-band bins (as the first
    version did, with a 4-term Blackman-Harris window and an even wider lobe)
    rejected clean breathing near the band centre. Until an estimate is
    accepted the output is 0.0; afterwards the last accepted value is held.
    """

    def __init__(self, samp_rate=500.0, window_sec=15.0, fft_size=32768,
                 f_low=0.1, f_high=0.5, snr_gate=6.0, min_pp=0.005):
        gr.sync_block.__init__(self, name='FFT BPM Estimator',
                               in_sig=[np.float32], out_sig=[np.float32])
        self.samp_rate = float(samp_rate)
        self.window_len = int(self.samp_rate * float(window_sec))
        # Never crop the record: pad to at least the next power of two above it.
        self.fft_size = max(int(fft_size), 1 << (self.window_len - 1).bit_length())
        self.snr_gate = float(snr_gate)
        self.min_pp = float(min_pp)
        self.buffer = np.zeros(self.window_len, dtype=np.float32)
        self.buf_idx = 0
        self.filled = False
        self.current_bpm = 0.0
        self.last_peak_ratio = 0.0     # diagnostic: peak / noise reference at the last update
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.samp_rate)
        self.band_mask = (self.freqs >= float(f_low)) & (self.freqs <= float(f_high))
        self.mainlobe_hz = 2.0 * self.samp_rate / self.window_len
        n_arr = np.arange(self.window_len, dtype=np.float64)
        self.win = (0.5 - 0.5 * np.cos(2.0 * np.pi * n_arr / (self.window_len - 1))).astype(np.float32)
        self.bin_hz = self.freqs[1] - self.freqs[0]
        self.detrend_t = np.arange(self.window_len, dtype=np.float32)
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
        if float(np.max(sig) - np.min(sig)) < self.min_pp:
            return
        coef = np.polyfit(self.detrend_t, sig, 1)
        sig = sig - (coef[0] * self.detrend_t + coef[1])
        spec = np.abs(np.fft.rfft(sig * self.win, n=self.fft_size)) ** 2
        band = np.where(self.band_mask, spec, 0.0)
        k = int(np.argmax(band))
        if k == 0 or k >= len(spec) - 1:
            return
        # Noise reference: in-band bins clear of the main lobe around the peak.
        offset = np.abs(self.freqs - self.freqs[k])
        ref = self.band_mask & (offset > self.mainlobe_hz)
        if np.count_nonzero(ref) < 3:
            ref = self.band_mask & (offset > 0.5 * self.mainlobe_hz)
        noise = float(np.median(spec[ref])) + 1e-30 if np.any(ref) else 1e-30
        self.last_peak_ratio = float(spec[k]) / noise
        if self.last_peak_ratio < self.snr_gate:
            return
        a = np.log(spec[k - 1] + 1e-30)
        b = np.log(spec[k] + 1e-30)
        c = np.log(spec[k + 1] + 1e-30)
        d = a - 2.0 * b + c
        delta = 0.5 * (a - c) / d if d != 0 else 0.0
        peak_freq = self.freqs[k] + delta * self.bin_hz
        self.current_bpm = float(np.clip(peak_freq * 60.0, 0.0, 60.0))

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
