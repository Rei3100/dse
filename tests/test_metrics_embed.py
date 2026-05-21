import os, sys, tempfile, datetime
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _make_test_flac(path: str) -> None:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3] * 2, axis=1).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def _make_before_after():
    from DSRE import MetricsComputer
    t = np.linspace(0, 1.0, SR, endpoint=False)
    mono = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    before = MetricsComputer.compute(mono, SR)
    after = MetricsComputer.compute(mono * 1.01, SR)
    return before, after


def test_embed_output_metadata_adds_dsre_version():
    from DSRE import _embed_output_metadata
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, [], before, after, level=5)
        f = FLAC(path)
        assert "dsre_version" in f
        assert "dsre_processed_utc" in f
        assert "dsre_level" in f
        assert f["dsre_level"][0] == "5"


def test_embed_output_metadata_adds_before_after_metrics():
    from DSRE import _embed_output_metadata
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, [], before, after, level=5)
        f = FLAC(path)
        assert "dsre_before_rms_db" in f
        assert "dsre_after_rms_db" in f
        assert "dsre_before_dr" in f
        assert "dsre_after_dr" in f


def test_embed_output_metadata_preserves_pictures():
    from DSRE import _embed_output_metadata, _extract_flac_pictures
    from mutagen.flac import FLAC, Picture
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        pic = Picture()
        pic.type = 3
        pic.mime = "image/jxl"
        pic.data = _FAKE_JXL
        f = FLAC(path)
        f.add_picture(pic)
        f.save()
        pictures = _extract_flac_pictures(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, pictures, before, after, level=5)
        result_pics = _extract_flac_pictures(path)
        assert len(result_pics) == 1
        assert result_pics[0].data == _FAKE_JXL
        assert result_pics[0].mime == "image/jxl"


def test_embed_output_metadata_no_crash_on_none_metrics():
    from DSRE import _embed_output_metadata
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        before = {k: None for k in ["rms_db", "peak_db", "dr", "plr", "lufs"]}
        after = {k: None for k in ["rms_db", "peak_db", "dr", "plr", "lufs"]}
        _embed_output_metadata(path, [], before, after, level=1)  # should not raise
