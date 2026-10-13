"""
Motion Detector — GNU Radio Embedded Python Block
===================================================
Place this after your Abs Phase Derivative block.
It compares the input against a threshold and outputs
1.0 (motion detected) or 0.0 (no motion).

Uses a short averaging window to avoid triggering on
single-sample noise spikes.

GNU Radio setup:
  Abs Phase Derivative ──► [this block] ──► QT GUI Time Sink
                                             (Sample Rate: 500)
                                         ──► QT GUI Number Sink
                                             (Sample Rate: 500)

Parameters:
  threshold   : phase derivative threshold in rad/sample (default 0.05)
                Start with 0.05 and adjust based on your setup.
                - Too low → triggers on noise
                - Too high → misses subtle motion
  avg_window  : number of samples to average over (default 50)
                At 500 samp/s, 50 samples = 100 ms averaging window.
                This smooths out single-sample noise spikes.
  hold_samples: how long to keep "motion detected" after the last
                trigger (default 500 = 1 second at 500 samp/s).
                Prevents rapid flickering between states.
"""
import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """Motion detector: thresholds abs phase derivative."""

    def __init__(self, threshold=0.05, avg_window=50, hold_samples=500):
        gr.sync_block.__init__(
            self,
            name="Motion Detector",
            in_sig=[np.float32],
            out_sig=[np.float32],
        )
        self.threshold = float(threshold)
        self.avg_window = int(avg_window)
        self.hold_samples = int(hold_samples)

        # Ring buffer for running average
        self.ring = np.zeros(self.avg_window, dtype=np.float32)
        self.ring_idx = 0
        self.ring_sum = 0.0

        # Hold timer: counts down from hold_samples after last trigger
        self.hold_counter = 0

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        n = len(x)

        for i in range(n):
            # Update running average
            old_val = self.ring[self.ring_idx]
            self.ring[self.ring_idx] = x[i]
            self.ring_sum += x[i] - old_val
            self.ring_idx = (self.ring_idx + 1) % self.avg_window

            avg = self.ring_sum / self.avg_window

            # Check threshold
            if avg > self.threshold:
                self.hold_counter = self.hold_samples
                out[i] = 1.0
            elif self.hold_counter > 0:
                self.hold_counter -= 1
                out[i] = 1.0
            else:
                out[i] = 0.0

        return n