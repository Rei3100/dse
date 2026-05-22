import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000

def test_normalize_volume_clip_zero():
    """正規化後 true peak は TP_TARGET_DBFS 以下、clip=0。"""
    from DSRE import _dsre_normalize_volume, TP_TARGET_DBFS, _true_peak
    t = np.linspace(0, 1.0, SR, endpoint=False)
    # (samples, 1) samples-first — save_flac24_out の data と同形状
    audio = (np.sin(2 * np.pi * 440 * t) * 5.0).astype(np.float32).reshape(-1, 1)
    out = _dsre_normalize_volume(audio, SR)
    tp_db = 20 * np.log10(_true_peak(out) + 1e-30)
    assert tp_db <= TP_TARGET_DBFS + 0.5
    assert int(np.sum(np.abs(out) >= 1.0)) == 0


def test_normalize_volume_legacy_mode():
    """DSRE_VOLUME_OPTIMIZE=0 で旧 sample-peak only モードに分岐。"""
    from DSRE import _dsre_normalize_volume
    t = np.linspace(0, 1.0, SR, endpoint=False)
    # (samples, 1) samples-first
    audio = (np.sin(2 * np.pi * 440 * t) * 1.5).astype(np.float32).reshape(-1, 1)
    os.environ["DSRE_VOLUME_OPTIMIZE"] = "0"
    try:
        out = _dsre_normalize_volume(audio, SR)
    finally:
        del os.environ["DSRE_VOLUME_OPTIMIZE"]
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-6
