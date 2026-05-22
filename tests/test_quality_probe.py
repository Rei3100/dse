import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_audio(freq=440.0, amp=0.3, dur=2.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def test_score_basic_signal():
    from DSRE import QualityProbe
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        sf.write(path, _mk_audio(amp=0.3), SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        assert result is not None
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.flagged, bool)
        assert isinstance(result.metrics, dict)


def test_score_clip_zero_after_normalize():
    """clip 大量入力でも正規化後の解析では clip_count = 0。"""
    from DSRE import QualityProbe
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        audio = _mk_audio(amp=5.0)  # clip 大量
        sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        assert result is not None
        assert result.metrics.get("clip_count", 0) == 0


def test_flag_brick_wall():
    """flatness が極小なら flagged=True。"""
    from DSRE import QualityProbe, _BRICK_WALL_FLATNESS
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        sf.write(path, _mk_audio(freq=440, amp=0.3, dur=2.0), SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        if result.metrics.get("flatness") is not None and result.metrics["flatness"] < _BRICK_WALL_FLATNESS:
            assert result.flagged
