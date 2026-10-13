import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    """Phase unwrap: removes +/-2*pi discontinuities"""
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Phase Unwrap',
            in_sig=[np.float32],
            out_sig=[np.float32])
        self.last_in = 0.0
        self.accum = 0.0

    def work(self, input_items, output_items):
        x = input_items[0]
        out = output_items[0]
        # running unwrap: track cumulative offset
        diff = np.diff(np.concatenate(([self.last_in], x)))
        # wrap each diff into [-pi, pi]
        diff_wrapped = (diff + np.pi) % (2*np.pi) - np.pi
        # reconstruct unwrapped signal
        unwrapped = self.accum + np.cumsum(diff_wrapped)
        out[:] = unwrapped.astype(np.float32)
        # update state for next call
        self.last_in = float(x[-1])
        self.accum = float(unwrapped[-1])
        return len(out)