"""Majority vote across the three BPM estimators."""

AGREE_TOL_BPM = 3.0
LABELS = ("FFT", "ZC", "AC")


def consensus(fft_bpm: float, zc_bpm: float, ac_bpm: float, tol: float = AGREE_TOL_BPM):
    """Combine three estimates into one reading.

    Returns ``(value, confidence)`` where ``value`` is the mean of the closest
    agreeing pair (or None) and ``confidence`` is one of:

    * ``"waiting"``   fewer than two estimators have produced a value
    * ``"3/3"``       all three are within ``tol`` of the consensus
    * ``"FFT+ZC"``, ``"FFT+AC"``, ``"ZC+AC"``  only that pair agrees
    * ``"diverging"`` no pair agrees within ``tol``

    An estimator that has not locked yet reports 0, which is treated as absent.
    """
    values = (fft_bpm, zc_bpm, ac_bpm)
    live = [i for i, v in enumerate(values) if v > 0.5]
    if len(live) < 2:
        return None, "waiting"

    pairs = sorted((abs(values[i] - values[j]), i, j)
                   for a, i in enumerate(live) for j in live[a + 1:])
    diff, i, j = pairs[0]
    if diff > tol:
        return None, "diverging"

    value = (values[i] + values[j]) / 2.0
    n_agree = sum(1 for k in live if abs(values[k] - value) <= tol)
    if n_agree == 3:
        return value, "3/3"
    return value, f"{LABELS[i]}+{LABELS[j]}"
