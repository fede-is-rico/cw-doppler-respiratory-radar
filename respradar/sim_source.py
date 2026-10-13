"""Hardware-free stand-in for the PlutoSDR receive stream.

Models what the radar sees: the 100 kHz IF tone reflected by a static
background plus a chest wall whose range changes with breathing, slow drift,
and occasional gross body movement. Two-way range change d(t) becomes a phase
shift of 4*pi*d/lambda radians, so an 8 mm peak-to-peak breath at 5.95 GHz is
about 2 rad of phase swing, which is why the chain needs a phase unwrap.

The block produces complex64 at the Pluto sample rate; feed it into the same
channeliser the hardware path uses.
"""
import numpy as np
from gnuradio import gr

from . import params as P


class breathing_sim_source(gr.sync_block):
    """Synthetic IF-tone echo from a breathing (and occasionally moving) target.

    Args:
        breath_bpm:        breathing rate.
        breath_amp_mm:     chest displacement amplitude (half of peak-to-peak).
        static_amp:        amplitude of the stationary echo (walls, antenna coupling).
        body_amp:          amplitude of the echo from the chest.
        noise_std:         complex AWGN standard deviation per 1 MS/s sample
                           (default 0.05, a clean constellation). The 2000:1
                           decimation gives ~33 dB of processing gain, so even
                           0.3 is still a clean signal at 500 S/s.
        drift_mm/drift_hz: slow range wander (posture, thermal drift) that the DC
                           blocker and bandpass have to reject.
        motion_period_s:   seconds between gross movements (0 disables them).
        motion_travel_mm:  how far the body shifts during a movement.
        motion_dur_s:      how long a movement takes (raised-cosine profile).
    """

    def __init__(self, samp_rate=P.SAMP_RATE, if_freq=P.IF_FREQ, rad_per_mm=P.RAD_PER_MM,
                 breath_bpm=15.0, breath_amp_mm=4.0,
                 static_amp=0.3, body_amp=0.5, noise_std=0.05,
                 drift_mm=1.0, drift_hz=0.01,
                 motion_period_s=0.0, motion_travel_mm=300.0, motion_dur_s=1.0,
                 seed=1):
        gr.sync_block.__init__(self, name="Breathing Sim Source",
                               in_sig=None, out_sig=[np.complex64])
        self.fs = float(samp_rate)
        self.if_cycles_per_sample = float(if_freq) / self.fs
        self.rad_per_mm = float(rad_per_mm)
        self.breath_hz = float(breath_bpm) / 60.0
        self.breath_amp_mm = float(breath_amp_mm)
        self.static_amp = float(static_amp)
        self.body_amp = float(body_amp)
        self.noise_std = float(noise_std)
        self.drift_mm = float(drift_mm)
        self.drift_hz = float(drift_hz)
        self.motion_period_s = float(motion_period_s)
        self.motion_travel_mm = float(motion_travel_mm)
        self.motion_dur_s = float(motion_dur_s)
        self.rng = np.random.default_rng(seed)
        self.n = 0

    def displacement_mm(self, t: np.ndarray) -> np.ndarray:
        """Two-way range change of the chest at times t (seconds)."""
        d = self.breath_amp_mm * np.sin(2.0 * np.pi * self.breath_hz * t)
        if self.drift_mm:
            d = d + self.drift_mm * np.sin(2.0 * np.pi * self.drift_hz * t)
        if self.motion_period_s > 0:
            k = np.floor(t / self.motion_period_s)          # movements completed
            tau = t - k * self.motion_period_s              # time into current period
            frac = np.clip(tau / self.motion_dur_s, 0.0, 1.0)
            d = d + self.motion_travel_mm * (k + 0.5 * (1.0 - np.cos(np.pi * frac)))
        return d

    def work(self, input_items, output_items):
        out = output_items[0]
        n = len(out)
        idx = self.n + np.arange(n, dtype=np.float64)
        t = idx / self.fs

        phi = self.rad_per_mm * self.displacement_mm(t)
        echo = self.static_amp + self.body_amp * np.exp(1j * phi)
        if_phase = 2.0 * np.pi * np.mod(idx * self.if_cycles_per_sample, 1.0)
        sig = echo * np.exp(1j * if_phase)
        if self.noise_std > 0:
            noise = self.rng.standard_normal(2 * n).view(np.complex128)
            sig = sig + (self.noise_std / np.sqrt(2.0)) * noise

        out[:] = sig.astype(np.complex64)
        self.n += n
        return n
