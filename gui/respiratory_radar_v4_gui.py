#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Respiratory Radar V4 — Custom Dark-Theme GUI
=============================================
A drop-in replacement for the GRC-generated Qt sinks.
Runs the full GNU Radio flowgraph internally and renders
every signal on a polished pyqtgraph dashboard.

Requirements (already present on any GNU Radio 3.10 + PlutoSDR machine):
    pip install pyqtgraph

Usage:
    python3 respiratory_radar_v4_gui.py

    Make sure all respiratory_radar_epy_block_*.py files are in the
    same directory (or on PYTHONPATH).

Signal Tap Architecture
-----------------------
Every data path from the GNU Radio flowgraph to the GUI passes through
a clearly-marked TAP POINT.  Each tap is a thread-safe ring buffer
(`TapBuffer`).  To add logging, ZMQ, or reroute a signal:

    1. Find the tap by name in the TAPS dict.
    2. Call  tap.latest(n)   to read the last n samples.
    3. Or replace the tap's consumer with a ZMQ pub, CSV writer, etc.

Search for "── TAP POINT" to find every injection site.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
import os
import csv
import signal
import time
import threading
import numpy as np

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

from gnuradio import gr, blocks, filter as gr_filter, analog, iio
from gnuradio.filter import firdes
from gnuradio.fft import window
from scipy.signal import firwin

# Embedded Python blocks — must be on PYTHONPATH or in CWD
import respiratory_radar_epy_block_0   as epy_block_0    # Phase Unwrap (breathing)
import respiratory_radar_epy_block_0_0 as epy_block_0_0  # Phase Unwrap (motion)
import respiratory_radar_epy_block_1   as epy_block_1    # Abs Phase Derivative
import respiratory_radar_epy_block_2   as epy_block_2    # Motion Detector
import respiratory_radar_epy_block_3   as epy_block_3    # FFT BPM Estimator
import respiratory_radar_epy_block_4   as epy_block_4    # ZC BPM Estimator
import respiratory_radar_epy_block_5   as epy_block_5    # AC BPM Estimator


# ─────────────────────────────────────────────────────────────────────────────
# Theme Configuration — edit these to restyle the whole GUI
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    # Background & surface
    "bg":               "#0B0E14",      # deep navy-black
    "surface":          "#111620",      # card / panel background
    "surface_border":   "#1E2738",      # subtle card border
    "grid":             "#1A2030",      # plot grid lines

    # Text
    "text_primary":     "#E6EDF3",      # headings, values
    "text_secondary":   "#7D8CA3",      # labels, captions
    "text_dim":         "#3D4A5C",      # disabled / placeholder

    # Accent palette
    "cyan":             "#00E5FF",      # breathing waveform, IQ dots
    "cyan_dim":         "#004D5A",      # IQ constellation grid
    "teal":             "#00BFA5",      # AC BPM
    "green":            "#00E676",      # consensus / "still" LED
    "amber":            "#FFD740",      # ZC BPM / warning
    "orange":           "#FF9100",      # FFT BPM
    "red":              "#FF1744",      # motion / alert
    "red_dim":          "#5A0015",      # motion fill
    "purple":           "#B388FF",      # phase derivative trace
    "magenta":          "#FF4081",      # threshold line

    # Fonts
    "font_family":      "Segoe UI, Roboto, Ubuntu, sans-serif",
    "font_bpm_size":    48,             # BPM big number
    "font_label_size":  11,             # panel labels
    "font_title_size":  13,             # section titles
    "font_status_size": 10,             # status bar
}

# ── Debug: set True to print data-flow stats every second to terminal ────────
DEBUG_DATAFLOW = True


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe ring buffer for signal taps
# ─────────────────────────────────────────────────────────────────────────────
class TapBuffer:
    """Thread-safe ring buffer for float32 samples.

    GNU Radio work() threads write via push(); the Qt GUI timer reads via
    latest().  The buffer is large enough that 20 Hz GUI reads never miss
    data at the expected sample rates.
    """
    def __init__(self, name: str, capacity: int = 50_000):
        self.name = name
        self.capacity = capacity
        self.buf = np.zeros(capacity, dtype=np.float32)
        self._widx = 0          # monotonically increasing total write count
        self._lock = threading.Lock()

    def push(self, data: np.ndarray):
        """Called from GNU Radio work() thread.  Handles data > capacity."""
        n = len(data)
        if n == 0:
            return
        with self._lock:
            # If chunk is bigger than the buffer, keep only the tail
            if n >= self.capacity:
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
        """Return the last *n* samples (or fewer if not yet available)."""
        with self._lock:
            total = self._widx
            available = min(n, total, self.capacity)
            if available == 0:
                return np.zeros(0, dtype=np.float32)
            end = total % self.capacity
            if end == 0:
                # Write pointer wrapped exactly to start → data at tail
                return self.buf[self.capacity - available:].copy()
            elif available <= end:
                return self.buf[end - available:end].copy()
            else:
                # Wraps around
                tail_start = self.capacity - (available - end)
                return np.concatenate([
                    self.buf[tail_start:],
                    self.buf[:end]
                ]).copy()

    @property
    def total_written(self):
        return self._widx


class TapBufferComplex:
    """Same as TapBuffer but for complex64 IQ samples."""
    def __init__(self, name: str, capacity: int = 10_000):
        self.name = name
        self.capacity = capacity
        self.buf = np.zeros(capacity, dtype=np.complex64)
        self._widx = 0
        self._lock = threading.Lock()

    def push(self, data: np.ndarray):
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
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
        with self._lock:
            total = self._widx
            available = min(n, total, self.capacity)
            if available == 0:
                return np.zeros(0, dtype=np.complex64)
            end = total % self.capacity
            if end == 0:
                return self.buf[self.capacity - available:].copy()
            elif available <= end:
                return self.buf[end - available:end].copy()
            else:
                tail_start = self.capacity - (available - end)
                return np.concatenate([
                    self.buf[tail_start:],
                    self.buf[:end]
                ]).copy()

    @property
    def total_written(self):
        return self._widx


# ─────────────────────────────────────────────────────────────────────────────
# GNU Radio probe sinks → push into TapBuffers
# ─────────────────────────────────────────────────────────────────────────────
class probe_sink_f(gr.sync_block):
    """Float32 probe that pushes every work() chunk into a TapBuffer."""
    def __init__(self, tap: TapBuffer):
        gr.sync_block.__init__(self, name="probe_f_" + tap.name,
                               in_sig=[np.float32], out_sig=[])
        self.tap = tap

    def work(self, input_items, output_items):
        self.tap.push(input_items[0].copy())
        return len(input_items[0])


class probe_sink_c(gr.sync_block):
    """Complex64 probe that pushes every work() chunk into a TapBufferComplex."""
    def __init__(self, tap: TapBufferComplex):
        gr.sync_block.__init__(self, name="probe_c_" + tap.name,
                               in_sig=[np.complex64], out_sig=[])
        self.tap = tap

    def work(self, input_items, output_items):
        self.tap.push(input_items[0].copy())
        return len(input_items[0])


# ─────────────────────────────────────────────────────────────────────────────
# Decimating probe for raw IQ (1 MSPS → keep every Nth sample for display)
# ─────────────────────────────────────────────────────────────────────────────
class probe_sink_c_decimated(gr.sync_block):
    """Complex64 probe that decimates before pushing (for raw IQ at 1 MSPS).

    At 1 MSPS the Pluto delivers ~32768 samples per work() call.  The IQ
    constellation only needs ~2000 points per frame at 20 Hz, so we keep
    every `decim`-th sample to avoid flooding the buffer.
    """
    def __init__(self, tap: TapBufferComplex, decim: int = 64):
        gr.sync_block.__init__(self, name="probe_c_dec_" + tap.name,
                               in_sig=[np.complex64], out_sig=[])
        self.tap = tap
        self.decim = max(1, int(decim))

    def work(self, input_items, output_items):
        data = input_items[0]
        # Take every Nth sample — cheap, no filtering needed for constellation
        decimated = data[::self.decim].copy()
        self.tap.push(decimated)
        return len(data)


# ─────────────────────────────────────────────────────────────────────────────
# All TAP POINTS — global dict so the GUI and flowgraph share them
# ─────────────────────────────────────────────────────────────────────────────
TAPS = {
    # ── TAP POINT: raw_iq ──  Raw IQ from PlutoSDR (pre-xlating), complex64
    #    Decimated 4:1 in the probe → ~250k SPS effective into buffer
    "raw_iq":           TapBufferComplex("raw_iq", capacity=50_000),

    # ── TAP POINT: breathing ──  Bandpass-filtered breathing waveform, float32
    #    500 SPS, 20s display = 10k samples
    "breathing":        TapBuffer("breathing", capacity=30_000),

    # ── TAP POINT: motion_raw ──  |dφ/dn| abs phase derivative, float32
    "motion_raw":       TapBuffer("motion_raw", capacity=15_000),

    # ── TAP POINT: motion_detect ──  Binary 1.0/0.0 motion flag, float32
    "motion_detect":    TapBuffer("motion_detect", capacity=15_000),

    # ── TAP POINT: bpm_fft ──  FFT BPM estimator output (constant within window)
    "bpm_fft":          TapBuffer("bpm_fft", capacity=4_000),

    # ── TAP POINT: bpm_zc ──  Zero-crossing BPM output
    "bpm_zc":           TapBuffer("bpm_zc", capacity=4_000),

    # ── TAP POINT: bpm_ac ──  Autocorrelation BPM output
    "bpm_ac":           TapBuffer("bpm_ac", capacity=4_000),

    # ── TAP POINT: phase_unwrapped_breathing ──  Unwrapped phase (breathing path)
    #    Not plotted by default, but available for debugging.
    "phase_unwrapped_breathing": TapBuffer("phase_unwrapped_breath", capacity=20_000),

    # ── TAP POINT: phase_unwrapped_motion ──  Unwrapped phase (motion path)
    #    Not plotted by default, but available for debugging.
    "phase_unwrapped_motion":    TapBuffer("phase_unwrapped_motion", capacity=20_000),
}


# ─────────────────────────────────────────────────────────────────────────────
# GNU Radio Flowgraph  (matches respiratory_radar_V4.grc exactly)
# ─────────────────────────────────────────────────────────────────────────────
class RespiratoryRadarFlowgraph(gr.top_block):
    """
    Identical signal chain to respiratory_radar_V4.grc, but every GUI sink
    is replaced with a probe_sink that feeds a TapBuffer.
    """

    def __init__(self):
        gr.top_block.__init__(self, "Respiratory Radar V4", catch_exceptions=True)

        # ── Variables ────────────────────────────────────────────────────────
        self.samp_rate   = 1_000_000
        self.center_freq = 5_950_000_000
        self.rx_gain     = 50
        self.tx_gain     = 0

        fs_dec = self.samp_rate / 2000          # = 500 SPS after xlating FIR
        self.fs_dec = fs_dec

        # ── RF front end ─────────────────────────────────────────────────────
        self.tx_tone = analog.sig_source_c(
            self.samp_rate, analog.GR_COS_WAVE, 100_000, 1, 0, 0)

        self.pluto_sink = iio.fmcomms2_sink_fc32(
            '192.168.2.1' if '192.168.2.1' else iio.get_pluto_uri(),
            [True, True], 32768, False)
        self.pluto_sink.set_len_tag_key('')
        self.pluto_sink.set_bandwidth(20_000_000)
        self.pluto_sink.set_frequency(self.center_freq)
        self.pluto_sink.set_samplerate(self.samp_rate)
        self.pluto_sink.set_attenuation(0, self.tx_gain)
        self.pluto_sink.set_filter_params('Auto', '', 0, 0)

        self.pluto_source = iio.fmcomms2_source_fc32(
            '192.168.2.1' if '192.168.2.1' else iio.get_pluto_uri(),
            [True, True], 32768)
        self.pluto_source.set_len_tag_key('packet_len')
        self.pluto_source.set_frequency(self.center_freq)
        self.pluto_source.set_samplerate(self.samp_rate)
        self.pluto_source.set_gain_mode(0, 'manual')
        self.pluto_source.set_gain(0, self.rx_gain)
        self.pluto_source.set_quadrature(True)
        self.pluto_source.set_rfdc(True)
        self.pluto_source.set_bbdc(True)
        self.pluto_source.set_filter_params('Auto', '', 0, 0)

        # ── Channelisation & decimation ──────────────────────────────────────
        self.xlating_fir = gr_filter.freq_xlating_fir_filter_ccc(
            2000,
            firdes.low_pass(1.0, self.samp_rate, 100, 50, window.WIN_HAMMING),
            100_000,
            self.samp_rate)

        # ── BREATHING PATH ───────────────────────────────────────────────────
        self.dc_blocker   = gr_filter.dc_blocker_cc(3200, True)
        self.c2arg_breath = blocks.complex_to_arg(1)
        self.unwrap_breath = epy_block_0.blk()
        self.bandpass     = gr_filter.fir_filter_fff(
            1, firwin(2001, [0.1, 0.5], pass_zero=False, fs=500))
        self.bandpass.declare_sample_delay(500)

        # BPM estimators
        self.fft_bpm = epy_block_3.blk(
            samp_rate=500, window_sec=15, fft_size=8192, f_low=0.1, f_high=0.5)
        self.zc_bpm  = epy_block_4.blk(
            samp_rate=500, window_sec=15.0, frac_of_pp=0.3, min_pp=0.005)
        self.ac_bpm  = epy_block_5.blk(
            samp_rate=500, window_sec=60.0, bpm_min=3.0, bpm_max=30.0,
            corr_threshold=0.3)

        # ── MOTION PATH (separate phase extraction, no DC blocker) ───────────
        self.c2arg_motion  = blocks.complex_to_arg(1)
        self.unwrap_motion = epy_block_0_0.blk()
        self.abs_dphi      = epy_block_1.blk(wrap_phase=False)
        self.motion_det    = epy_block_2.blk(
            threshold=0.1, avg_window=50, hold_samples=500)

        # ── PROBE SINKS (→ TapBuffers) ───────────────────────────────────────
        # ── TAP POINT: raw_iq ── (decim=1 → full 1 MSPS, matches GRC qtgui_const_sink)
        # 500 buffered samples = 0.5 ms = ~50 rotations of the 100 kHz IF tone
        self.probe_raw_iq     = probe_sink_c_decimated(TAPS["raw_iq"], decim=1)

        # ── TAP POINT: breathing ──
        self.probe_breathing  = probe_sink_f(TAPS["breathing"])

        # ── TAP POINT: motion_raw ──
        self.probe_motion_raw = probe_sink_f(TAPS["motion_raw"])

        # ── TAP POINT: motion_detect ──
        self.probe_motion_det = probe_sink_f(TAPS["motion_detect"])

        # ── TAP POINT: bpm_fft ──
        self.probe_bpm_fft    = probe_sink_f(TAPS["bpm_fft"])

        # ── TAP POINT: bpm_zc ──
        self.probe_bpm_zc     = probe_sink_f(TAPS["bpm_zc"])

        # ── TAP POINT: bpm_ac ──
        self.probe_bpm_ac     = probe_sink_f(TAPS["bpm_ac"])

        # ── TAP POINT: phase_unwrapped_breathing ──
        self.probe_phase_breath = probe_sink_f(TAPS["phase_unwrapped_breathing"])

        # ── TAP POINT: phase_unwrapped_motion ──
        self.probe_phase_motion = probe_sink_f(TAPS["phase_unwrapped_motion"])

        # ── Connections (mirrors V4.grc exactly) ─────────────────────────────
        # TX tone → Pluto TX
        self.connect((self.tx_tone, 0),         (self.pluto_sink, 0))

        # Pluto RX → xlating FIR
        self.connect((self.pluto_source, 0),    (self.xlating_fir, 0))

        # Pluto RX → Raw IQ tap (decimated probe)
        self.connect((self.pluto_source, 0),    (self.probe_raw_iq, 0))

        # Xlating → DC blocker → breathing path
        self.connect((self.xlating_fir, 0),     (self.dc_blocker, 0))
        self.connect((self.dc_blocker, 0),      (self.c2arg_breath, 0))
        self.connect((self.c2arg_breath, 0),    (self.unwrap_breath, 0))
        self.connect((self.unwrap_breath, 0),   (self.bandpass, 0))
        self.connect((self.unwrap_breath, 0),   (self.probe_phase_breath, 0))

        # Bandpass → BPM estimators + breathing tap
        self.connect((self.bandpass, 0),        (self.fft_bpm, 0))
        self.connect((self.bandpass, 0),        (self.zc_bpm, 0))
        self.connect((self.bandpass, 0),        (self.ac_bpm, 0))
        self.connect((self.bandpass, 0),        (self.probe_breathing, 0))

        # BPM estimators → taps
        self.connect((self.fft_bpm, 0),         (self.probe_bpm_fft, 0))
        self.connect((self.zc_bpm, 0),          (self.probe_bpm_zc, 0))
        self.connect((self.ac_bpm, 0),          (self.probe_bpm_ac, 0))

        # Xlating → motion path (bypasses DC blocker)
        self.connect((self.xlating_fir, 0),     (self.c2arg_motion, 0))
        self.connect((self.c2arg_motion, 0),    (self.unwrap_motion, 0))
        self.connect((self.unwrap_motion, 0),   (self.abs_dphi, 0))
        self.connect((self.unwrap_motion, 0),   (self.probe_phase_motion, 0))
        self.connect((self.abs_dphi, 0),        (self.motion_det, 0))
        self.connect((self.abs_dphi, 0),        (self.probe_motion_raw, 0))
        self.connect((self.motion_det, 0),      (self.probe_motion_det, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: styled plot widget factory
# ─────────────────────────────────────────────────────────────────────────────
def make_plot(title="", x_label="", y_label="", y_range=None, square=False):
    """Create a pyqtgraph PlotWidget with the dark theme applied."""
    pw = pg.PlotWidget()
    pw.setBackground(THEME["surface"])
    pw.showGrid(x=True, y=True, alpha=0.15)
    pw.getAxis('bottom').setPen(pg.mkPen(THEME["text_dim"], width=1))
    pw.getAxis('left').setPen(pg.mkPen(THEME["text_dim"], width=1))
    pw.getAxis('bottom').setTextPen(pg.mkPen(THEME["text_secondary"]))
    pw.getAxis('left').setTextPen(pg.mkPen(THEME["text_secondary"]))

    if title:
        pw.setTitle(title, color=THEME["text_primary"],
                    size=f"{THEME['font_title_size']}pt")
    if x_label:
        pw.setLabel('bottom', x_label, color=THEME["text_secondary"])
    if y_label:
        pw.setLabel('left', y_label, color=THEME["text_secondary"])
    if y_range:
        pw.setYRange(*y_range)
    if square:
        pw.getViewBox().setAspectLocked(True)

    pw.setStyleSheet(f"""
        border: 1px solid {THEME['surface_border']};
        border-radius: 6px;
    """)
    return pw


# ─────────────────────────────────────────────────────────────────────────────
# BPM gauge widget — large coloured number + method label
# ─────────────────────────────────────────────────────────────────────────────
class BpmGauge(QtWidgets.QFrame):
    def __init__(self, method_name: str, accent_color: str, parent=None):
        super().__init__(parent)
        self.accent = accent_color
        self._value = 0.0

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['surface']};
                border: 1px solid {THEME['surface_border']};
                border-radius: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self.label = QtWidgets.QLabel(method_name)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setStyleSheet(f"""
            color: {THEME['text_secondary']};
            font-size: {THEME['font_label_size']}pt;
            font-weight: 600; border: none;
        """)
        layout.addWidget(self.label)

        self.number = QtWidgets.QLabel("--")
        self.number.setAlignment(QtCore.Qt.AlignCenter)
        self._set_number_style(THEME['text_dim'])
        layout.addWidget(self.number)

        unit = QtWidgets.QLabel("BPM")
        unit.setAlignment(QtCore.Qt.AlignCenter)
        unit.setStyleSheet(f"""
            color: {THEME['text_dim']};
            font-size: {THEME['font_label_size'] - 1}pt;
            border: none;
        """)
        layout.addWidget(unit)

    def _set_number_style(self, color):
        self.number.setStyleSheet(f"""
            color: {color};
            font-size: {THEME['font_bpm_size']}pt;
            font-weight: 700;
            font-family: {THEME['font_family']};
            border: none;
        """)

    def set_value(self, v: float):
        self._value = v
        if v < 0.5:
            self.number.setText("--")
            self._set_number_style(THEME['text_dim'])
        else:
            self.number.setText(f"{v:.1f}")
            self._set_number_style(self.accent)

    @property
    def value(self):
        return self._value


# ─────────────────────────────────────────────────────────────────────────────
# Motion LED indicator
# ─────────────────────────────────────────────────────────────────────────────
class MotionLed(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 180)
        self._active = False
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['surface']};
                border: 1px solid {THEME['surface_border']};
                border-radius: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(6)

        self.dot = QtWidgets.QLabel()
        self.dot.setFixedSize(60, 60)
        self.dot.setAlignment(QtCore.Qt.AlignCenter)
        self._set_dot(False)
        layout.addWidget(self.dot, 0, QtCore.Qt.AlignCenter)

        self.status_label = QtWidgets.QLabel("STILL")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet(f"""
            color: {THEME['green']};
            font-size: 14pt; font-weight: 700; border: none;
        """)
        layout.addWidget(self.status_label)

        cap = QtWidgets.QLabel("MOTION")
        cap.setAlignment(QtCore.Qt.AlignCenter)
        cap.setStyleSheet(f"""
            color: {THEME['text_dim']};
            font-size: {THEME['font_label_size'] - 1}pt;
            border: none;
        """)
        layout.addWidget(cap)

    def _set_dot(self, active: bool):
        color = THEME['red'] if active else THEME['green']
        self.dot.setStyleSheet(f"""
            background-color: {color};
            border-radius: 30px;
            border: 3px solid {color}44;
        """)

    def set_motion(self, active: bool):
        if active == self._active:
            return
        self._active = active
        self._set_dot(active)
        if active:
            self.status_label.setText("MOTION")
            self.status_label.setStyleSheet(f"""
                color: {THEME['red']};
                font-size: 14pt; font-weight: 700; border: none;
            """)
        else:
            self.status_label.setText("STILL")
            self.status_label.setStyleSheet(f"""
                color: {THEME['green']};
                font-size: 14pt; font-weight: 700; border: none;
            """)


# ─────────────────────────────────────────────────────────────────────────────
# Consensus BPM display
# ─────────────────────────────────────────────────────────────────────────────
class ConsensusBpm(QtWidgets.QFrame):
    AGREE_TOL = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)

        # Public attributes for SessionLogger to read
        self.value = None             # float consensus BPM, or None if unavailable
        self.confidence = "waiting"   # "3/3", "FFT+ZC", "diverging", "waiting"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['surface']};
                border: 1px solid {THEME['surface_border']};
                border-radius: 10px;
            }}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(0)

        lbl = QtWidgets.QLabel("CONSENSUS")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet(f"""
            color: {THEME['text_secondary']};
            font-size: {THEME['font_label_size']}pt;
            font-weight: 600; border: none;
        """)
        layout.addWidget(lbl)

        self.number = QtWidgets.QLabel("--")
        self.number.setAlignment(QtCore.Qt.AlignCenter)
        self._set_style(THEME['text_dim'])
        layout.addWidget(self.number)

        self.conf_label = QtWidgets.QLabel("")
        self.conf_label.setAlignment(QtCore.Qt.AlignCenter)
        self.conf_label.setStyleSheet(f"""
            color: {THEME['text_dim']}; font-size: 9pt; border: none;
        """)
        layout.addWidget(self.conf_label)

    def _set_style(self, color):
        self.number.setStyleSheet(f"""
            color: {color}; font-size: 36pt;
            font-weight: 700; font-family: {THEME['font_family']}; border: none;
        """)

    def update_from(self, fft_v, zc_v, ac_v):
        vals = [v for v in (fft_v, zc_v, ac_v) if v > 0.5]
        if len(vals) < 2:
            self.value = None
            self.confidence = "waiting"
            self.number.setText("--")
            self._set_style(THEME['text_dim'])
            self.conf_label.setText("waiting…")
            return

        labels = ["FFT", "ZC", "AC"]
        all_v = [fft_v, zc_v, ac_v]
        pairs = []
        for i in range(3):
            for j in range(i+1, 3):
                if all_v[i] > 0.5 and all_v[j] > 0.5:
                    pairs.append((abs(all_v[i] - all_v[j]), i, j))
        pairs.sort()
        best_diff, i, j = pairs[0]

        if best_diff <= self.AGREE_TOL:
            med = (all_v[i] + all_v[j]) / 2.0
            self.value = med
            self.number.setText(f"{med:.1f}")
            n_agree = sum(1 for v in vals if abs(v - med) <= self.AGREE_TOL)
            if n_agree == 3:
                color = THEME['green']
                conf = "3/3 agree"
                self.confidence = "3/3"
            else:
                color = THEME['amber']
                conf = f"{labels[i]}+{labels[j]} agree"
                self.confidence = f"{labels[i]}+{labels[j]}"
            self._set_style(color)
            self.conf_label.setText(conf)
            self.conf_label.setStyleSheet(f"""
                color: {color}88; font-size: 9pt; border: none;
            """)
        else:
            self.value = None
            self.confidence = "diverging"
            self.number.setText("??")
            self._set_style(THEME['red'])
            self.conf_label.setText("diverging")
            self.conf_label.setStyleSheet(f"""
                color: {THEME['red']}88; font-size: 9pt; border: none;
            """)


# ─────────────────────────────────────────────────────────────────────────────
# CSV Session Logger
# ─────────────────────────────────────────────────────────────────────────────
class SessionLogger:
    """Logs BPM samples and motion events to a timestamped CSV file.

    Schema:
        t_sec, event, fft_bpm, zc_bpm, ac_bpm,
        consensus_bpm, consensus_confidence, motion_state, notes

    Event types:
        SESSION_START  — first row of a recording, notes = filename
        BPM_SAMPLE     — periodic snapshot (default every 1 s)
        MOTION_START   — rising edge of motion detector
        MOTION_END     — falling edge of motion detector
        SESSION_END    — last row before file close

    `t_sec` is monotonic-clock seconds since the session started — immune to
    wall-clock changes (NTP, manual edits) during recording.
    """

    HEADER = ["t_sec", "event", "fft_bpm", "zc_bpm", "ac_bpm",
              "consensus_bpm", "consensus_confidence", "motion_state", "notes"]

    def __init__(self):
        self.file = None
        self.writer = None
        self.start_mono = None
        self.path = None
        self.events_logged = 0

    @property
    def is_recording(self):
        return self.file is not None

    def start(self, directory="."):
        """Open a new CSV file, write the header, log SESSION_START."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"radar_session_{ts}.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.start_mono = time.monotonic()
        self.events_logged = 0
        wall = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write("SESSION_START",
                    notes=f"{os.path.basename(self.path)} | wall_start={wall}")
        return self.path

    def stop(self):
        """Log SESSION_END and close the file."""
        if self.file is not None:
            self._write("SESSION_END")
            self.file.close()
            self.file = None
            self.writer = None
            self.start_mono = None

    def t_elapsed(self):
        if self.start_mono is None:
            return 0.0
        return time.monotonic() - self.start_mono

    def _write(self, event_type, fft=None, zc=None, ac=None,
               consensus=None, confidence=None, motion=None, notes=""):
        if self.writer is None:
            return
        row = [
            f"{self.t_elapsed():.3f}",
            event_type,
            f"{fft:.2f}" if fft is not None else "",
            f"{zc:.2f}" if zc is not None else "",
            f"{ac:.2f}" if ac is not None else "",
            f"{consensus:.2f}" if consensus is not None else "",
            confidence if confidence is not None else "",
            f"{int(motion)}" if motion is not None else "",
            notes,
        ]
        self.writer.writerow(row)
        self.file.flush()           # survive a crash — every row hits disk
        self.events_logged += 1

    def log_bpm_sample(self, fft, zc, ac, consensus, confidence, motion):
        """Periodic BPM snapshot — call every 1 s during recording."""
        self._write("BPM_SAMPLE",
                    fft=fft, zc=zc, ac=ac,
                    consensus=consensus, confidence=confidence,
                    motion=motion)

    def log_motion_event(self, started, fft, zc, ac, consensus, confidence):
        """Rising/falling edge of the motion detector — call on transition only."""
        ev = "MOTION_START" if started else "MOTION_END"
        self._write(ev,
                    fft=fft, zc=zc, ac=ac,
                    consensus=consensus, confidence=confidence,
                    motion=(1 if started else 0))


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI Window
# ─────────────────────────────────────────────────────────────────────────────
class RadarDashboard(QtWidgets.QMainWindow):
    REFRESH_MS = 50   # 20 Hz GUI refresh — responsive yet light on CPU

    def __init__(self, flowgraph: RespiratoryRadarFlowgraph):
        super().__init__()
        self.fg = flowgraph
        self.start_time = time.time()
        self._debug_counter = 0

        # ── Session logger state ─────────────────────────────────────────────
        self.logger = SessionLogger()
        self._last_motion_state = 0     # for edge detection
        self._bpm_sample_counter = 0    # refresh ticks since last BPM_SAMPLE

        self.setWindowTitle("Respiratory Radar V4 — Dashboard")
        self.setMinimumSize(1400, 900)

        # Global dark stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {THEME['bg']};
            }}
            QWidget {{
                background-color: {THEME['bg']};
                color: {THEME['text_primary']};
                font-family: {THEME['font_family']};
            }}
            QStatusBar {{
                background-color: {THEME['surface']};
                color: {THEME['text_secondary']};
                font-size: {THEME['font_status_size']}pt;
                border-top: 1px solid {THEME['surface_border']};
            }}
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 6)
        root.setSpacing(10)

        # ── Title bar ────────────────────────────────────────────────────────
        title_bar = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("  RESPIRATORY RADAR")
        title_lbl.setStyleSheet(f"""
            color: {THEME['cyan']};
            font-size: 18pt; font-weight: 800; letter-spacing: 2px;
        """)
        title_bar.addWidget(title_lbl)

        subtitle = QtWidgets.QLabel(
            "V4  ·  5.95 GHz CW Doppler  ·  PlutoSDR")
        subtitle.setStyleSheet(f"""
            color: {THEME['text_dim']};
            font-size: 11pt; padding-top: 6px;
        """)
        title_bar.addWidget(subtitle)
        title_bar.addStretch()

        # ── Record button ────────────────────────────────────────────────────
        self.record_btn = QtWidgets.QPushButton("● START RECORDING")
        self.record_btn.setCheckable(True)
        self.record_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.record_btn.setMinimumHeight(36)
        self.record_btn.setMinimumWidth(200)
        self.record_btn.clicked.connect(self._toggle_recording)
        self._apply_record_btn_idle_style()
        title_bar.addWidget(self.record_btn)

        root.addLayout(title_bar)

        # ── Row 1: IQ + BPM gauges + Motion LED ─────────────────────────────
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)

        # IQ constellation (square) — matches GRC qtgui_const_sink_x style:
        #   style=0 (solid line connecting samples) + marker=0 (circle at each sample)
        #   X axis ±2, Y axis ±0.15, autoscale OFF, 500-sample buffer
        self.iq_plot = make_plot(
            title="RAW IQ CONSTELLATION",
            x_label="I", y_label="Q", square=True)
        self.iq_plot.setFixedWidth(320)
        self.iq_plot.setFixedHeight(320)
        self.iq_plot.disableAutoRange()
        self.iq_plot.setXRange(-1.0, 1.0)
        self.iq_plot.setYRange(-0.1, 0.1)

        # Circle markers at each sample (no connecting line)
        self.iq_scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(THEME["cyan"]),
            size=3, pxMode=True)
        self.iq_plot.addItem(self.iq_scatter)
        row1.addWidget(self.iq_plot)

        # BPM gauges column
        bpm_col = QtWidgets.QVBoxLayout()
        bpm_col.setSpacing(8)
        bpm_row = QtWidgets.QHBoxLayout()
        bpm_row.setSpacing(8)
        self.gauge_fft = BpmGauge("FFT", THEME["orange"])
        self.gauge_zc  = BpmGauge("ZC",  THEME["amber"])
        self.gauge_ac  = BpmGauge("AC",  THEME["teal"])
        bpm_row.addWidget(self.gauge_fft)
        bpm_row.addWidget(self.gauge_zc)
        bpm_row.addWidget(self.gauge_ac)
        bpm_col.addLayout(bpm_row)
        self.consensus = ConsensusBpm()
        bpm_col.addWidget(self.consensus)
        row1.addLayout(bpm_col, stretch=1)

        # Motion LED
        self.motion_led = MotionLed()
        row1.addWidget(self.motion_led)
        root.addLayout(row1)

        # ── Row 2: Breathing waveform (full width) ──────────────────────────
        self.breath_plot = make_plot(
            title="BREATHING WAVEFORM  (20 s)",
            x_label="Time (s)", y_label="Phase (rad)")
        self.breath_plot.setMinimumHeight(200)
        self.breath_plot.enableAutoRange(axis='y')
        self.breath_curve = self.breath_plot.plot(
            pen=pg.mkPen(THEME["cyan"], width=2))
        root.addWidget(self.breath_plot, stretch=2)

        # ── Row 3: Motion raw + Motion binary ────────────────────────────────
        row3 = QtWidgets.QHBoxLayout()
        row3.setSpacing(10)

        # Motion raw |dφ/dn|
        self.motion_raw_plot = make_plot(
            title="|dφ/dn|  RAW PHASE RATE",
            x_label="Time (s)", y_label="rad/sample")
        self.motion_raw_plot.setMinimumHeight(160)
        self.motion_raw_plot.enableAutoRange(axis='y')
        self.motion_raw_curve = self.motion_raw_plot.plot(
            pen=pg.mkPen(THEME["purple"], width=1.5))
        self.thresh_line = pg.InfiniteLine(
            pos=0.1, angle=0,
            pen=pg.mkPen(THEME["magenta"], width=1.5,
                         style=QtCore.Qt.DashLine),
            label="thresh=0.1",
            labelOpts={"color": THEME["magenta"], "position": 0.95})
        self.motion_raw_plot.addItem(self.thresh_line)
        row3.addWidget(self.motion_raw_plot, stretch=1)

        # Motion binary — PlotCurveItem with fill under, reused every frame
        self.motion_bin_plot = make_plot(
            title="MOTION DETECTED",
            x_label="Time (s)", y_label="",
            y_range=(-0.1, 1.3))
        self.motion_bin_plot.setMinimumHeight(160)
        self.motion_bin_curve = pg.PlotCurveItem(
            pen=pg.mkPen(THEME["red"], width=2),
            fillLevel=0.0,
            brush=pg.mkBrush(THEME["red_dim"]))
        self.motion_bin_plot.addItem(self.motion_bin_curve)
        row3.addWidget(self.motion_bin_plot, stretch=1)

        root.addLayout(row3, stretch=1)

        # ── Status bar ──────────────────────────────────────────────────────
        self.status = self.statusBar()
        self.status.showMessage("Waiting for data from PlutoSDR…")

        # ── Refresh timer ───────────────────────────────────────────────────
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._refresh)
        self.timer.start(self.REFRESH_MS)

    # ─────────────────────────────────────────────────────────────────────────
    # Record button styling & toggle
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_record_btn_idle_style(self):
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['surface']};
                color: {THEME['red']};
                border: 1px solid {THEME['red']};
                border-radius: 6px;
                font-size: 11pt; font-weight: 700; letter-spacing: 1px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: {THEME['red_dim']};
            }}
        """)

    def _apply_record_btn_active_style(self):
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['red']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['red']};
                border-radius: 6px;
                font-size: 11pt; font-weight: 700; letter-spacing: 1px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: #FF4D5C;
            }}
        """)

    def _toggle_recording(self):
        if self.record_btn.isChecked():
            try:
                path = self.logger.start()
                self.record_btn.setText("■ STOP RECORDING")
                self._apply_record_btn_active_style()
                # Reset edge-detector and sample counter for the new session
                self._last_motion_state = 0
                self._bpm_sample_counter = 0
                print(f"[LOGGER] Recording started → {path}")
            except OSError as e:
                self.record_btn.setChecked(False)
                self.status.showMessage(f"Failed to open log file: {e}", 5000)
        else:
            if self.logger.is_recording:
                path = self.logger.path
                n = self.logger.events_logged
                self.logger.stop()
                print(f"[LOGGER] Recording stopped → {path} ({n} events)")
            self.record_btn.setText("● START RECORDING")
            self._apply_record_btn_idle_style()

    # ─────────────────────────────────────────────────────────────────────────
    def _refresh(self):
        """Called ~20× per second.  Pulls from TapBuffers → updates widgets."""
        fs = 500.0

        # ── Debug: print data flow stats once per second ─────────────────────
        self._debug_counter += 1
        if DEBUG_DATAFLOW and (self._debug_counter % 20 == 0):
            stats = {name: tap.total_written for name, tap in TAPS.items()}
            print(f"[DATAFLOW] {stats}")

        # ── IQ constellation ─────────────────────────────────────────────────
        # Pull 500 latest samples to match GRC qtgui_const_sink size=500.
        # At 1 MSPS that's 0.5 ms = ~50 rotations of the 100 kHz IF tone.
        iq = TAPS["raw_iq"].latest(500)
        if len(iq) > 10:
            self.iq_scatter.setData(iq.real, iq.imag)

        # ── Breathing waveform ───────────────────────────────────────────────
        breath = TAPS["breathing"].latest(10_000)
        if len(breath) > 10:
            t = np.arange(len(breath)) / fs
            self.breath_curve.setData(t, breath)

        # ── BPM values (average last 50 samples to smooth jitter) ────────────
        fft_chunk = TAPS["bpm_fft"].latest(50)
        zc_chunk  = TAPS["bpm_zc"].latest(50)
        ac_chunk  = TAPS["bpm_ac"].latest(50)

        fft_v = float(np.mean(fft_chunk)) if len(fft_chunk) > 0 else 0.0
        zc_v  = float(np.mean(zc_chunk))  if len(zc_chunk)  > 0 else 0.0
        ac_v  = float(np.mean(ac_chunk))  if len(ac_chunk)  > 0 else 0.0

        self.gauge_fft.set_value(fft_v)
        self.gauge_zc.set_value(zc_v)
        self.gauge_ac.set_value(ac_v)
        self.consensus.update_from(fft_v, zc_v, ac_v)

        # ── Motion raw ───────────────────────────────────────────────────────
        motion_raw = TAPS["motion_raw"].latest(5_000)
        if len(motion_raw) > 10:
            t = np.arange(len(motion_raw)) / fs
            self.motion_raw_curve.setData(t, motion_raw)

        # ── Motion binary (reuse curve, just update data) ────────────────────
        motion_bin = TAPS["motion_detect"].latest(5_000)
        if len(motion_bin) > 10:
            t = np.arange(len(motion_bin)) / fs
            self.motion_bin_curve.setData(t, motion_bin)

        # ── Motion LED ───────────────────────────────────────────────────────
        det = TAPS["motion_detect"].latest(250)
        motion_now = 0
        if len(det) > 0:
            motion_now = int(float(np.max(det)) > 0.5)
            self.motion_led.set_motion(bool(motion_now))

        # ── Session logging ──────────────────────────────────────────────────
        if self.logger.is_recording:
            # 1. Motion edge detection — log only on transitions
            if motion_now != self._last_motion_state:
                self.logger.log_motion_event(
                    started=(motion_now == 1),
                    fft=fft_v, zc=zc_v, ac=ac_v,
                    consensus=self.consensus.value,
                    confidence=self.consensus.confidence)
                self._last_motion_state = motion_now

            # 2. Periodic BPM snapshot every 1 s (20 refresh ticks @ 20 Hz)
            self._bpm_sample_counter += 1
            if self._bpm_sample_counter >= 20:
                self._bpm_sample_counter = 0
                self.logger.log_bpm_sample(
                    fft=fft_v, zc=zc_v, ac=ac_v,
                    consensus=self.consensus.value,
                    confidence=self.consensus.confidence,
                    motion=motion_now)

        # ── Status bar ───────────────────────────────────────────────────────
        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        total_iq = TAPS["raw_iq"].total_written
        total_breath = TAPS["breathing"].total_written

        if self.logger.is_recording:
            rec_t = int(self.logger.t_elapsed())
            rs = rec_t % 60
            rm_total = rec_t // 60
            rm = rm_total % 60
            rh = rm_total // 60
            self.status.showMessage(
                f"  ● REC {rh:02d}:{rm:02d}:{rs:02d}   │   "
                f"events: {self.logger.events_logged:,}   │   "
                f"file: {os.path.basename(self.logger.path)}   │   "
                f"Fs: 500 SPS   │   "
                f"RF: 5.950 GHz")
        else:
            self.status.showMessage(
                f"  Uptime: {h:02d}:{m:02d}:{s:02d}   │   "
                f"IQ: {total_iq:,}   │   "
                f"Breath: {total_breath:,}   │   "
                f"Fs: 500 SPS   │   "
                f"RF: 5.950 GHz   │   "
                f"RX gain: 50 dB")

    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.timer.stop()
        if self.logger.is_recording:
            self.logger.stop()
        self.fg.stop()
        self.fg.wait()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # PyQt application — MUST be created first
    app = QtWidgets.QApplication(sys.argv)

    # pyqtgraph global defaults
    pg.setConfigOptions(
        background=THEME["bg"],
        foreground=THEME["text_primary"],
        antialias=True,
        useOpenGL=False,       # set True if you have GPU — much smoother
    )

    # Build flowgraph (don't start yet — let GUI appear first)
    print("[INIT] Building GNU Radio flowgraph…")
    fg = RespiratoryRadarFlowgraph()

    # Build and show dashboard
    print("[INIT] Building dashboard…")
    win = RadarDashboard(fg)
    win.show()

    # Let Qt paint the window before data starts flowing
    app.processEvents()

    # NOW start the flowgraph — data begins flowing into TapBuffers
    print("[INIT] Starting flowgraph — waiting for PlutoSDR data…")
    print("[INIT] Watch [DATAFLOW] lines below — numbers should increase.")
    print("[INIT] If all counts stay 0, the PlutoSDR is not connected.\n")
    fg.start()

    # Ctrl-C handler
    def sig_handler(sig=None, frame=None):
        print("\n[SHUTDOWN] Stopping flowgraph…")
        fg.stop()
        fg.wait()
        app.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Timer to keep event loop responsive to Ctrl-C
    keepalive = QtCore.QTimer()
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
