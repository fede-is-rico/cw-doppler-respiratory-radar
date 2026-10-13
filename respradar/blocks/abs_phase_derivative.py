import numpy as np
from gnuradio import gr


class blk(gr.sync_block):
    """
    Absolute phase derivative block.

    Input:
        float32 phase samples in radians

    Output:
        float32 |dphi/dn|

    Use:
        Connect this block's output to a QT GUI Time Sink to plot it.
    """

    def __init__(self, wrap_phase=True):
        super().__init__(
            name="Abs Phase Derivative",
            in_sig=[np.float32],
            out_sig=[np.float32],
        )
        self.wrap_phase = bool(wrap_phase)
        self.last = 0.0

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]

        n = len(x)
        if n == 0:
            return 0

        dphi = np.diff(np.concatenate(([self.last], x)))

        if self.wrap_phase:
            dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi

        out[:n] = np.abs(dphi).astype(np.float32)

        self.last = float(x[-1])
        return n