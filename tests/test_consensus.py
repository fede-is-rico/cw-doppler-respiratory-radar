from respradar.consensus import consensus


def test_waiting_until_two_estimators_report():
    assert consensus(0, 0, 0) == (None, "waiting")
    assert consensus(15, 0, 0) == (None, "waiting")


def test_all_three_agree():
    value, conf = consensus(15.0, 16.0, 15.5)
    assert conf == "3/3" and abs(value - 15.25) < 1e-9


def test_closest_pair_wins_and_is_named():
    value, conf = consensus(15.0, 22.0, 16.0)
    assert conf == "FFT+AC" and value == 15.5
    assert consensus(15.0, 15.5, 0)[1] == "FFT+ZC"
    assert consensus(0, 15.0, 15.5)[1] == "ZC+AC"


def test_diverging_when_no_pair_within_tolerance():
    assert consensus(15.0, 22.0, 30.0) == (None, "diverging")


def test_tolerance_is_inclusive():
    assert consensus(15.0, 18.0, 0)[1] == "FFT+ZC"
    assert consensus(15.0, 18.01, 0)[1] == "diverging"
