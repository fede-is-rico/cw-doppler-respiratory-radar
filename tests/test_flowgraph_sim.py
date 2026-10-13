"""End-to-end: simulator -> channeliser -> both paths -> estimators, headless.

Each run pushes about 80 s of 1 MS/s signal through the real flowgraph and
takes roughly 10 s of CPU.
"""
import numpy as np
import pytest

from respradar import params as P
from respradar.flowgraph import RespiratoryRadarFlowgraph


def run_sim(seconds, **sim_kwargs):
    fg = RespiratoryRadarFlowgraph(source="sim", sim_speed=None, head_seconds=seconds,
                                   sim_kwargs=sim_kwargs)
    fg.run()
    return fg


def test_all_three_estimators_lock_on_simulated_breathing():
    fg = run_sim(80, breath_bpm=15.0, breath_amp_mm=4.0)
    t = fg.taps
    fs = int(P.FS_DEC)
    assert t["breathing"].total_written >= 79 * fs
    for name in ("bpm_fft", "bpm_zc", "bpm_ac"):
        assert t[name].latest(1)[0] == pytest.approx(15.0, abs=1.0), name
    assert np.all(t["motion_detect"].latest(60 * fs) == 0.0)


def test_body_movement_trips_motion_detector_only_during_the_event():
    fg = run_sim(40, breath_bpm=12.0, motion_period_s=25.0, motion_travel_mm=300.0, motion_dur_s=1.0)
    fs = int(P.FS_DEC)
    tap = fg.taps["motion_detect"]
    det = tap.latest(tap.capacity)                       # the last 30 s (capacity-limited)
    t_end = tap.total_written / fs                       # ~40 s
    t = t_end - (len(det) - np.arange(len(det))) / fs    # time of each sample
    during = det[(t > 25.2) & (t < 26.8)]                # movement spans 25-26 s, hold adds 1 s
    quiet = det[(t > 15.0) & (t < 24.5)]
    assert during.min() == 1.0
    assert quiet.max() == 0.0
    on = t[det > 0.5]
    assert on.max() - on.min() == pytest.approx(1.0 + P.MOTION_HOLD_SAMPLES / fs, abs=0.3)
