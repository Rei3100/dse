import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000

def _sine(freq=1000.0, dur=2.0, amp=0.5):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)

def _stereo(mono):
    return np.stack([mono, mono], axis=0)


def test_metrics_keys_present():
    from DSRE import MetricsComputer
    audio = _sine()
    result = MetricsComputer.compute(audio, SR)
    expected_keys = [
        "rms_db", "peak_db", "dr", "plr", "lufs", "lra",
        "clip_count", "centroid_hz", "rolloff_hz", "flatness",
        "hf_ratio_4k", "hf_ratio_8k", "hf_ratio_12k", "hf_ratio_16k",
        "thd_proxy",
    ]
    for k in expected_keys:
        assert k in result, f"missing key: {k}"


def test_rms_db_reasonable():
    from DSRE import MetricsComputer
    audio = _sine(amp=0.5)
    result = MetricsComputer.compute(audio, SR)
    # 0.5 amplitude sine → RMS ≈ -9 dBFS
    assert -15 < result["rms_db"] < -5


def test_clip_count_zero_for_clean():
    from DSRE import MetricsComputer
    audio = _sine(amp=0.5)
    result = MetricsComputer.compute(audio, SR)
    assert result["clip_count"] == 0


def test_clip_count_nonzero_for_clipped():
    from DSRE import MetricsComputer
    audio = np.ones(SR, dtype=np.float32)  # 全サンプルが 1.0
    result = MetricsComputer.compute(audio, SR)
    assert result["clip_count"] > 0


def test_hf_ratio_4k_increases_with_hf_content():
    from DSRE import MetricsComputer
    lo = _sine(freq=500.0)
    hi = _sine(freq=8000.0)
    r_lo = MetricsComputer.compute(lo, SR)
    r_hi = MetricsComputer.compute(hi, SR)
    assert r_hi["hf_ratio_4k"] > r_lo["hf_ratio_4k"]


def test_stereo_handled():
    from DSRE import MetricsComputer
    audio = _stereo(_sine())
    result = MetricsComputer.compute(audio, SR)
    assert result["rms_db"] is not None
