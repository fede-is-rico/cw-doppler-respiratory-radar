#!/usr/bin/env python3
"""Respiratory radar dashboard.

Runs the V4 flowgraph (hardware or simulator) and renders every signal on a
dark pyqtgraph dashboard: raw IQ constellation, three BPM gauges with a
consensus vote, the breathing waveform, the motion-path phase rate, the motion
detector output, and a CSV session recorder.

    python -m respradar.dashboard                 # PlutoSDR at 192.168.2.1
    python -m respradar.dashboard --uri ip:192.168.2.1
    python -m respradar.dashboard --sim           # no hardware needed
    python -m respradar.dashboard --sim --sim-bpm 20 --sim-motion-period 30

Signal tap architecture: every path from GNU Radio to the GUI goes through a
TapBuffer (respradar/taps.py). GNU Radio work() threads push into them; the
20 Hz Qt timer in RadarDashboard._refresh reads the latest samples.
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from . import params as P
from .consensus import AGREE_TOL_BPM, consensus
from .flowgraph import RespiratoryRadarFlowgraph
from .session_logger import SessionLogger


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
    AGREE_TOL = AGREE_TOL_BPM

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

    def _set_conf(self, text, color):
        self.conf_label.setText(text)
        self.conf_label.setStyleSheet(f"color: {color}; font-size: 9pt; border: none;")

    def update_from(self, fft_v, zc_v, ac_v):
        self.value, self.confidence = consensus(fft_v, zc_v, ac_v, self.AGREE_TOL)
        if self.confidence == "waiting":
            self.number.setText("--")
            self._set_style(THEME['text_dim'])
            self._set_conf("waiting…", THEME['text_dim'])
        elif self.confidence == "diverging":
            self.number.setText("??")
            self._set_style(THEME['red'])
            self._set_conf("diverging", THEME['red'] + "88")
        else:
            color = THEME['green'] if self.confidence == "3/3" else THEME['amber']
            self.number.setText(f"{self.value:.1f}")
            self._set_style(color)
            self._set_conf(self.confidence + " agree", color + "88")


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI Window
# ─────────────────────────────────────────────────────────────────────────────
class RadarDashboard(QtWidgets.QMainWindow):
    REFRESH_MS = 50   # 20 Hz GUI refresh — responsive yet light on CPU

    def __init__(self, flowgraph: RespiratoryRadarFlowgraph, debug: bool = False,
                 log_dir: str = "."):
        super().__init__()
        self.fg = flowgraph
        self.taps = flowgraph.taps
        self.debug = debug
        self.log_dir = log_dir
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
            f"V4  ·  {P.CENTER_FREQ / 1e9:.2f} GHz CW Doppler  ·  "
            + ("SIMULATED TARGET" if flowgraph.source_kind == "sim" else "PlutoSDR"))
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
            pos=P.MOTION_THRESHOLD, angle=0,
            pen=pg.mkPen(THEME["magenta"], width=1.5,
                         style=QtCore.Qt.DashLine),
            label=f"thresh={P.MOTION_THRESHOLD:g}",
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
        self.status.showMessage("Waiting for data…")

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
                path = self.logger.start(self.log_dir)
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
        fs = P.FS_DEC

        # ── Debug: print data flow stats once per second ─────────────────────
        self._debug_counter += 1
        if self.debug and (self._debug_counter % 20 == 0):
            stats = {name: tap.total_written for name, tap in self.taps.items()}
            print(f"[DATAFLOW] {stats}")

        # ── IQ constellation ─────────────────────────────────────────────────
        # Pull 500 latest samples to match GRC qtgui_const_sink size=500.
        # At 1 MSPS that's 0.5 ms = ~50 rotations of the 100 kHz IF tone.
        iq = self.taps["raw_iq"].latest(500)
        if len(iq) > 10:
            self.iq_scatter.setData(iq.real, iq.imag)

        # ── Breathing waveform ───────────────────────────────────────────────
        breath = self.taps["breathing"].latest(10_000)
        if len(breath) > 10:
            t = np.arange(len(breath)) / fs
            self.breath_curve.setData(t, breath)

        # ── BPM values (average last 50 samples to smooth jitter) ────────────
        fft_chunk = self.taps["bpm_fft"].latest(50)
        zc_chunk  = self.taps["bpm_zc"].latest(50)
        ac_chunk  = self.taps["bpm_ac"].latest(50)

        fft_v = float(np.mean(fft_chunk)) if len(fft_chunk) > 0 else 0.0
        zc_v  = float(np.mean(zc_chunk))  if len(zc_chunk)  > 0 else 0.0
        ac_v  = float(np.mean(ac_chunk))  if len(ac_chunk)  > 0 else 0.0

        self.gauge_fft.set_value(fft_v)
        self.gauge_zc.set_value(zc_v)
        self.gauge_ac.set_value(ac_v)
        self.consensus.update_from(fft_v, zc_v, ac_v)

        # ── Motion raw ───────────────────────────────────────────────────────
        motion_raw = self.taps["motion_raw"].latest(5_000)
        if len(motion_raw) > 10:
            t = np.arange(len(motion_raw)) / fs
            self.motion_raw_curve.setData(t, motion_raw)

        # ── Motion binary (reuse curve, just update data) ────────────────────
        motion_bin = self.taps["motion_detect"].latest(5_000)
        if len(motion_bin) > 10:
            t = np.arange(len(motion_bin)) / fs
            self.motion_bin_curve.setData(t, motion_bin)

        # ── Motion LED ───────────────────────────────────────────────────────
        det = self.taps["motion_detect"].latest(250)
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
        total_iq = self.taps["raw_iq"].total_written
        total_breath = self.taps["breathing"].total_written

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
                f"Fs: {P.FS_DEC:g} SPS   │   "
                f"{self._source_text()}")
        else:
            self.status.showMessage(
                f"  Uptime: {h:02d}:{m:02d}:{s:02d}   │   "
                f"IQ: {total_iq:,}   │   "
                f"Breath: {total_breath:,}   │   "
                f"Fs: {P.FS_DEC:g} SPS   │   "
                f"{self._source_text()}")

    def _source_text(self):
        if self.fg.source_kind == "sim":
            return "Source: simulator"
        return f"RF: {P.CENTER_FREQ / 1e9:.3f} GHz   │   RX gain: {P.RX_GAIN_DB} dB"

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
def build_arg_parser():
    p = argparse.ArgumentParser(description="CW Doppler respiratory radar dashboard")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--uri", default=P.PLUTO_URI,
                     help="PlutoSDR address or libiio URI (default: %(default)s)")
    src.add_argument("--sim", action="store_true",
                     help="use the built-in breathing simulator instead of hardware")
    p.add_argument("--sim-bpm", type=float, default=15.0, help="simulated breathing rate")
    p.add_argument("--sim-amp-mm", type=float, default=4.0, help="simulated chest displacement amplitude")
    p.add_argument("--sim-motion-period", type=float, default=0.0,
                   help="seconds between simulated body movements (0 = none)")
    p.add_argument("--sim-speed", type=float, default=1.0,
                   help="simulator real-time factor (>1 fills the estimators faster)")
    p.add_argument("--log-dir", default=".", help="where session CSV files are written")
    p.add_argument("--debug", action="store_true", help="print tap-buffer sample counts once per second")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(background=THEME["bg"], foreground=THEME["text_primary"],
                        antialias=True, useOpenGL=False)

    print("[INIT] Building GNU Radio flowgraph…")
    if args.sim:
        fg = RespiratoryRadarFlowgraph(
            source="sim", sim_speed=args.sim_speed,
            sim_kwargs=dict(breath_bpm=args.sim_bpm, breath_amp_mm=args.sim_amp_mm,
                            motion_period_s=args.sim_motion_period))
    else:
        fg = RespiratoryRadarFlowgraph(source="pluto", uri=args.uri)

    print("[INIT] Building dashboard…")
    win = RadarDashboard(fg, debug=args.debug, log_dir=args.log_dir)
    win.show()
    app.processEvents()          # paint the window before data starts flowing

    print("[INIT] Starting flowgraph…")
    if not args.sim:
        print("[INIT] If the sample counts in the status bar stay at 0, the PlutoSDR is not connected.")
    fg.start()

    def sig_handler(sig=None, frame=None):
        print("\n[SHUTDOWN] Stopping flowgraph…")
        fg.stop()
        fg.wait()
        app.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    keepalive = QtCore.QTimer()          # lets Python see Ctrl-C while Qt runs
    keepalive.start(200)
    keepalive.timeout.connect(lambda: None)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
