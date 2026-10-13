"""Top-level GNU Radio flowgraph: the signal chain of grc/respiratory_radar_V4.grc
with every Qt sink replaced by a probe into a TapBuffer, plus a simulated front end.

    PlutoSDR / simulator (1 MS/s, IF tone at 100 kHz)
        └─ freq_xlating_fir_filter (shift 100 kHz -> 0, 100 Hz LPF, 2000:1 -> 500 S/s)
             ├─ breathing path: dc_blocker -> complex_to_arg -> phase unwrap
             │       -> 0.1-0.5 Hz bandpass -> FFT / zero-crossing / autocorrelation BPM
             └─ motion path:    complex_to_arg -> phase unwrap -> |dphi/dn| -> motion detector
"""
from gnuradio import analog, blocks, filter as gr_filter, gr
from gnuradio.fft import window
from gnuradio.filter import firdes
from scipy.signal import firwin

from . import params as P
from .blocks import abs_phase_derivative, ac_bpm, fft_bpm, motion_detector, phase_unwrap, zc_bpm
from .sim_source import breathing_sim_source
from .taps import make_taps, probe_sink


class RespiratoryRadarFlowgraph(gr.top_block):
    """Build with ``source="pluto"`` for hardware or ``source="sim"`` for the simulator.

    Args:
        uri:          libiio URI or address of the Pluto (hardware only).
        sim_kwargs:   keyword arguments for :class:`breathing_sim_source`.
        sim_speed:    real-time factor for the simulator (1.0 = real time);
                      None removes the throttle so a bounded run finishes as fast
                      as the CPU allows.
        head_seconds: stop the simulator after this many seconds of signal
                      (lets ``run()`` return, for tests).
        taps:         dict from :func:`make_taps`; a fresh one is made if omitted.
    """

    def __init__(self, source="pluto", uri=P.PLUTO_URI, sim_kwargs=None,
                 sim_speed=1.0, head_seconds=None, taps=None):
        gr.top_block.__init__(self, "CW Doppler Respiratory Radar V4", catch_exceptions=True)
        self.source_kind = source
        self.taps = taps if taps is not None else make_taps()

        # ── RF front end ─────────────────────────────────────────────────────
        if source == "pluto":
            from gnuradio import iio   # only needed with hardware
            uri = uri or iio.get_pluto_uri()
            self.tx_tone = analog.sig_source_c(P.SAMP_RATE, analog.GR_COS_WAVE, P.IF_FREQ, 1, 0, 0)
            self.pluto_sink = iio.fmcomms2_sink_fc32(uri, [True, True], P.PLUTO_BUFFER, False)
            self.pluto_sink.set_len_tag_key('')
            self.pluto_sink.set_bandwidth(P.RF_BANDWIDTH)
            self.pluto_sink.set_frequency(P.CENTER_FREQ)
            self.pluto_sink.set_samplerate(P.SAMP_RATE)
            self.pluto_sink.set_attenuation(0, P.TX_ATTEN_DB)
            self.pluto_sink.set_filter_params('Auto', '', 0, 0)

            self.pluto_source = iio.fmcomms2_source_fc32(uri, [True, True], P.PLUTO_BUFFER)
            self.pluto_source.set_len_tag_key('packet_len')
            self.pluto_source.set_frequency(P.CENTER_FREQ)
            self.pluto_source.set_samplerate(P.SAMP_RATE)
            self.pluto_source.set_gain_mode(0, 'manual')
            self.pluto_source.set_gain(0, P.RX_GAIN_DB)
            self.pluto_source.set_quadrature(True)
            self.pluto_source.set_rfdc(True)
            self.pluto_source.set_bbdc(True)
            self.pluto_source.set_filter_params('Auto', '', 0, 0)

            self.connect((self.tx_tone, 0), (self.pluto_sink, 0))
            rx = self.pluto_source
        elif source == "sim":
            self.sim = breathing_sim_source(**(sim_kwargs or {}))
            front = [self.sim]
            if sim_speed:
                self.throttle = blocks.throttle(gr.sizeof_gr_complex, P.SAMP_RATE * float(sim_speed), True)
                front.append(self.throttle)
            if head_seconds:
                self.head = blocks.head(gr.sizeof_gr_complex, int(head_seconds * P.SAMP_RATE))
                front.append(self.head)
            if not sim_speed or sim_speed >= 10:
                # Faster than real time the 2000:1 decimator would otherwise hand the
                # Python blocks downstream two samples per work() call. Bigger buffers
                # in front of it mean ~500 samples per call instead (0.25 s bursts at
                # real-time speed, so leave the default when pacing for the display).
                for b in front:
                    b.set_min_output_buffer(1 << 20)
            for a, b in zip(front, front[1:]):
                self.connect((a, 0), (b, 0))
            rx = front[-1]
        else:
            raise ValueError(f"source must be 'pluto' or 'sim', not {source!r}")
        self.rx = rx

        # ── Channelisation: IF tone -> baseband, 2000:1 to 500 S/s ───────────
        self.xlating_fir = gr_filter.freq_xlating_fir_filter_ccc(
            P.DECIM,
            firdes.low_pass(1.0, P.SAMP_RATE, P.XLATE_CUTOFF_HZ, P.XLATE_TRANSITION_HZ, window.WIN_HAMMING),
            P.IF_FREQ, P.SAMP_RATE)

        # ── Breathing path ───────────────────────────────────────────────────
        self.dc_blocker = gr_filter.dc_blocker_cc(P.DC_BLOCK_LEN, True)
        self.c2arg_breath = blocks.complex_to_arg(1)
        self.unwrap_breath = phase_unwrap.blk()
        self.bandpass = gr_filter.fir_filter_fff(
            1, firwin(P.BP_TAPS, [P.BP_LOW_HZ, P.BP_HIGH_HZ], pass_zero=False, fs=P.FS_DEC))
        self.bandpass.declare_sample_delay((P.BP_TAPS - 1) // 2)

        self.fft_bpm = fft_bpm.blk(samp_rate=P.FS_DEC, window_sec=P.FFT_WINDOW_SEC,
                                   fft_size=P.FFT_SIZE, f_low=P.BP_LOW_HZ, f_high=P.BP_HIGH_HZ,
                                   snr_gate=P.FFT_SNR_GATE, min_pp=P.MIN_PP_RAD)
        self.zc_bpm = zc_bpm.blk(samp_rate=P.FS_DEC, window_sec=P.ZC_WINDOW_SEC,
                                 frac_of_pp=P.ZC_FRAC_OF_PP, min_pp=P.MIN_PP_RAD)
        self.ac_bpm = ac_bpm.blk(samp_rate=P.FS_DEC, window_sec=P.AC_WINDOW_SEC,
                                 bpm_min=P.AC_BPM_MIN, bpm_max=P.AC_BPM_MAX,
                                 corr_threshold=P.AC_CORR_THRESHOLD)

        # ── Motion path (no DC blocker: gross motion must not be filtered out) ─
        self.c2arg_motion = blocks.complex_to_arg(1)
        self.unwrap_motion = phase_unwrap.blk()
        self.abs_dphi = abs_phase_derivative.blk(wrap_phase=False)
        self.motion_det = motion_detector.blk(threshold=P.MOTION_THRESHOLD,
                                              avg_window=P.MOTION_AVG_WINDOW,
                                              hold_samples=P.MOTION_HOLD_SAMPLES)

        # ── Probes into the tap buffers ──────────────────────────────────────
        t = self.taps
        self.probe_raw_iq = probe_sink(t["raw_iq"])
        self.probe_breathing = probe_sink(t["breathing"])
        self.probe_phase_breath = probe_sink(t["phase_breath"])
        self.probe_motion_raw = probe_sink(t["motion_raw"])
        self.probe_motion_det = probe_sink(t["motion_detect"])
        self.probe_phase_motion = probe_sink(t["phase_motion"])
        self.probe_bpm_fft = probe_sink(t["bpm_fft"])
        self.probe_bpm_zc = probe_sink(t["bpm_zc"])
        self.probe_bpm_ac = probe_sink(t["bpm_ac"])

        # ── Connections ──────────────────────────────────────────────────────
        self.connect((rx, 0), (self.xlating_fir, 0))
        self.connect((rx, 0), (self.probe_raw_iq, 0))

        self.connect((self.xlating_fir, 0), (self.dc_blocker, 0))
        self.connect((self.dc_blocker, 0), (self.c2arg_breath, 0))
        self.connect((self.c2arg_breath, 0), (self.unwrap_breath, 0))
        self.connect((self.unwrap_breath, 0), (self.bandpass, 0))
        self.connect((self.unwrap_breath, 0), (self.probe_phase_breath, 0))
        for est, probe in ((self.fft_bpm, self.probe_bpm_fft),
                           (self.zc_bpm, self.probe_bpm_zc),
                           (self.ac_bpm, self.probe_bpm_ac)):
            self.connect((self.bandpass, 0), (est, 0))
            self.connect((est, 0), (probe, 0))
        self.connect((self.bandpass, 0), (self.probe_breathing, 0))

        self.connect((self.xlating_fir, 0), (self.c2arg_motion, 0))
        self.connect((self.c2arg_motion, 0), (self.unwrap_motion, 0))
        self.connect((self.unwrap_motion, 0), (self.abs_dphi, 0))
        self.connect((self.unwrap_motion, 0), (self.probe_phase_motion, 0))
        self.connect((self.abs_dphi, 0), (self.motion_det, 0))
        self.connect((self.abs_dphi, 0), (self.probe_motion_raw, 0))
        self.connect((self.motion_det, 0), (self.probe_motion_det, 0))
