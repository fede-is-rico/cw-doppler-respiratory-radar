"""Phase unwrap, |dphi/dn|, and the motion detector."""
import numpy as np
import pytest

from respradar import params as P
from respradar.blocks import abs_phase_derivative, motion_detector, phase_unwrap
from tests.helpers import FS, run_block


def test_phase_unwrap_removes_2pi_jumps_across_chunks():
    t = np.arange(20 * FS) / FS
    true_phase = 3.0 * np.sin(2 * np.pi * 0.25 * t) + 0.5 * t     # swings past +/- pi and drifts
    wrapped = np.angle(np.exp(1j * true_phase)).astype(np.float32)
    out = run_block(phase_unwrap.blk(), wrapped, chunk=333)        # odd chunk size on purpose
    # the block starts from zero; compare up to the constant offset fixed by the first sample
    assert np.allclose(out - out[0], true_phase - true_phase[0], atol=1e-3)


def test_abs_phase_derivative_matches_numpy_diff():
    x = np.cumsum(np.random.default_rng(0).standard_normal(5000) * 0.01).astype(np.float32)
    out = run_block(abs_phase_derivative.blk(wrap_phase=False), x, chunk=777)
    expect = np.abs(np.diff(np.concatenate(([0.0], x))))
    assert np.allclose(out, expect, atol=1e-6)


def make_detector():
    return motion_detector.blk(threshold=P.MOTION_THRESHOLD, avg_window=P.MOTION_AVG_WINDOW,
                               hold_samples=P.MOTION_HOLD_SAMPLES)


def test_motion_detector_ignores_breathing_rate_phase_change():
    """Breathing at 2 rad peak-to-peak gives |dphi/dn| ~ 0.003 rad/sample: far below the gate."""
    t = np.arange(30 * FS) / FS
    phase = 1.0 * np.sin(2 * np.pi * 0.3 * t)
    dphi = np.abs(np.diff(np.concatenate(([0.0], phase)))).astype(np.float32)
    assert np.all(run_block(make_detector(), dphi) == 0.0)


def test_motion_detector_fires_on_body_movement_and_holds():
    """A 30 cm shift in one second is ~0.23 rad/sample: trips the gate, then holds for 1 s."""
    fs = int(FS)
    dphi = np.zeros(10 * fs, dtype=np.float32)
    dphi[3 * fs:4 * fs] = 0.23
    out = run_block(make_detector(), dphi, chunk=250)
    on = np.flatnonzero(out > 0.5)
    assert on[0] == pytest.approx(3 * fs, abs=P.MOTION_AVG_WINDOW)      # fires once the average crosses
    assert on[-1] == pytest.approx(4 * fs + P.MOTION_HOLD_SAMPLES, abs=P.MOTION_AVG_WINDOW)
    assert np.all(out[on[0]:on[-1]] == 1.0)                            # no flicker inside the event


def test_motion_detector_rejects_single_sample_spikes():
    fs = int(FS)
    dphi = np.zeros(5 * fs, dtype=np.float32)
    dphi[1000] = 2.0                 # one glitch: average over 50 samples is 0.04 < 0.1
    assert np.all(run_block(make_detector(), dphi) == 0.0)
