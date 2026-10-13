"""Radar and signal-chain parameters shared by the flowgraph, simulator, and dashboard.

These mirror the values in grc/respiratory_radar_V4.grc. Change them here and
there together.
"""
import math

# ── RF front end (ADALM-Pluto, AD9363) ───────────────────────────────────────
SAMP_RATE = 1_000_000          # S/s at the Pluto
CENTER_FREQ = 5_950_000_000    # Hz
IF_FREQ = 100_000              # Hz: TX tone offset, so the echo sits off DC
RF_BANDWIDTH = 20_000_000      # Hz: AD9363 analog filter
RX_GAIN_DB = 50
TX_ATTEN_DB = 0
PLUTO_BUFFER = 32768
PLUTO_URI = "192.168.2.1"      # gr-iio also accepts "ip:192.168.2.1" or "usb:"

# ── Channelisation ───────────────────────────────────────────────────────────
DECIM = 2000                   # 1 MS/s -> 500 S/s
FS_DEC = SAMP_RATE / DECIM     # 500 S/s
XLATE_CUTOFF_HZ = 100
XLATE_TRANSITION_HZ = 50

# ── Breathing path ───────────────────────────────────────────────────────────
DC_BLOCK_LEN = 3200            # samples at 500 S/s (6.4 s), removes static clutter
BP_TAPS = 2001                 # 4 s FIR, 0.1-0.5 Hz = 6-30 BPM
BP_LOW_HZ = 0.1
BP_HIGH_HZ = 0.5

FFT_WINDOW_SEC = 15
FFT_SIZE = 8192
ZC_WINDOW_SEC = 15.0
ZC_FRAC_OF_PP = 0.3
MIN_PP_RAD = 0.005
AC_WINDOW_SEC = 60.0
AC_BPM_MIN = 3.0
AC_BPM_MAX = 30.0
AC_CORR_THRESHOLD = 0.3

# ── Motion path ──────────────────────────────────────────────────────────────
MOTION_THRESHOLD = 0.1         # rad/sample of |dphi/dn| at 500 S/s
MOTION_AVG_WINDOW = 50         # samples (100 ms)
MOTION_HOLD_SAMPLES = 500      # samples (1 s)

# ── Derived geometry ─────────────────────────────────────────────────────────
C = 299_792_458.0
WAVELENGTH_M = C / CENTER_FREQ                      # 50.4 mm
MM_PER_RAD = WAVELENGTH_M * 1000.0 / (4.0 * math.pi)  # 4.01 mm of range per radian (two-way)
RAD_PER_MM = 1.0 / MM_PER_RAD
