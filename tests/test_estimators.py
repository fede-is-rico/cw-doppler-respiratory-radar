"""Each BPM estimator on synthetic breathing, with the flowgraph's parameters."""
import numpy as np
import pytest

from respradar import params as P
from respradar.blocks import ac_bpm, fft_bpm, zc_bpm
from tests.helpers import FS, bandpassed_noise, breathing, run_block

RATES = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 27.0]


def make_fft():
    return fft_bpm.blk(samp_rate=FS, window_sec=P.FFT_WINDOW_SEC, fft_size=P.FFT_SIZE,
                       f_low=P.BP_LOW_HZ, f_high=P.BP_HIGH_HZ,
                       snr_gate=P.FFT_SNR_GATE, min_pp=P.MIN_PP_RAD)


def make_zc():
    return zc_bpm.blk(samp_rate=FS, window_sec=P.ZC_WINDOW_SEC,
                      frac_of_pp=P.ZC_FRAC_OF_PP, min_pp=P.MIN_PP_RAD)


def make_ac():
    return ac_bpm.blk(samp_rate=FS, window_sec=P.AC_WINDOW_SEC, bpm_min=P.AC_BPM_MIN,
                      bpm_max=P.AC_BPM_MAX, corr_threshold=P.AC_CORR_THRESHOLD)


@pytest.mark.parametrize("bpm", RATES)
def test_fft_tracks_clean_breathing(bpm):
    """The deployed 15 s window must accept every rate in the band, not just some."""
    blk = make_fft()
    out = run_block(blk, breathing(bpm, 3 * P.FFT_WINDOW_SEC))
    assert out[-1] == pytest.approx(bpm, abs=0.3), f"gate ratio {blk.last_peak_ratio:.1f}"


@pytest.mark.parametrize("bpm", RATES)
def test_zc_tracks_clean_breathing(bpm):
    out = run_block(make_zc(), breathing(bpm, 3 * P.ZC_WINDOW_SEC))
    assert out[-1] == pytest.approx(bpm, abs=0.3)


@pytest.mark.parametrize("bpm", RATES)
def test_ac_tracks_clean_breathing(bpm):
    out = run_block(make_ac(), breathing(bpm, P.AC_WINDOW_SEC + 1))
    assert out[-1] == pytest.approx(bpm, abs=0.3)


def test_fft_gate_margin_is_comfortable_across_band():
    """Peak / noise-reference ratio stays well above the gate for every rate."""
    for bpm in RATES:
        blk = make_fft()
        run_block(blk, breathing(bpm, 2 * P.FFT_WINDOW_SEC))
        assert blk.last_peak_ratio > 5 * P.FFT_SNR_GATE, (bpm, blk.last_peak_ratio)


def test_fft_gate_rejects_most_noise_only_windows():
    """Filtered noise alone must usually be rejected.

    A 15 s record has only ~6 independent spectral samples across 0.1-0.5 Hz,
    so no in-band SNR gate can reject noise every time: measured false-accept
    rate at the deployed gate is about 40%. Presence detection belongs on the
    DC-blocked IQ magnitude, not here (see docs/DESIGN_NOTES.md).
    """
    accepted = 0
    windows = 0
    L = int(P.FFT_WINDOW_SEC * FS)
    for seed in range(40):
        blk = make_fft()
        x = bandpassed_noise(4 * P.FFT_WINDOW_SEC, std=0.05, seed=seed)
        for w in range(4):
            run_block(blk, x[w * L:(w + 1) * L])
            windows += 1
            accepted += blk.last_peak_ratio >= P.FFT_SNR_GATE
    assert accepted / windows < 0.5


def test_fft_low_rate_bias_is_small():
    """Two cycles in a 15 s record: the Hann main lobe touches 0.1 Hz and the
    estimate biases low by a few percent. Documented, not hidden."""
    out = run_block(make_fft(), breathing(8.0, 3 * P.FFT_WINDOW_SEC))
    assert out[-1] == pytest.approx(8.0, abs=0.5)


def test_estimators_hold_zero_for_tiny_signal():
    """Below min_pp nothing is reported (ZC and FFT share that floor)."""
    tiny = breathing(15.0, 40, amp=0.4 * P.MIN_PP_RAD)
    assert np.all(run_block(make_zc(), tiny) == 0.0)
    assert np.all(run_block(make_fft(), tiny) == 0.0)


def test_fft_output_is_zero_until_window_fills():
    out = run_block(make_fft(), breathing(15.0, 2 * P.FFT_WINDOW_SEC), chunk=100)
    first = int(P.FFT_WINDOW_SEC * FS)
    assert np.all(out[:first - 100] == 0.0)
    assert out[first + 100] == pytest.approx(15.0, abs=0.3)


def test_fft_does_not_crop_a_window_longer_than_fft_size():
    """20 s at 500 S/s is 10000 samples; a 8192-point FFT must zero-pad, not truncate."""
    blk = fft_bpm.blk(samp_rate=FS, window_sec=20.0, fft_size=8192)
    assert blk.fft_size >= blk.window_len
    out = run_block(blk, breathing(20.0, 45))
    assert out[-1] == pytest.approx(20.0, abs=0.3)


@pytest.mark.parametrize("make", [make_fft, make_zc, make_ac])
def test_blocks_survive_chunks_larger_than_their_window(make):
    """GNU Radio may hand a block more samples than its window in one work() call."""
    blk = make()
    big = int(1.5 * max(P.FFT_WINDOW_SEC, P.AC_WINDOW_SEC) * FS)
    out = run_block(blk, breathing(15.0, 150), chunk=big)
    assert out[-1] == pytest.approx(15.0, abs=0.5)


def test_fft_reports_after_signal_resumes_following_noise():
    """Gate closes on noise and the next clean window is accepted again."""
    blk = make_fft()
    x = np.concatenate([bandpassed_noise(2 * P.FFT_WINDOW_SEC, std=0.05, seed=1),
                        breathing(12.0, 3 * P.FFT_WINDOW_SEC)])
    out = run_block(blk, x)
    assert out[-1] == pytest.approx(12.0, abs=0.3)
