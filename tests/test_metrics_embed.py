import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _make_test_flac(path: str) -> None:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3] * 2, axis=1).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def test_embed_output_metadata_adds_dsre_tags():
    from DSRE import _embed_output_metadata
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        _embed_output_metadata(path, [])
        f = FLAC(path)
        assert "dsre_version" in f
        assert "dsre_processed_utc" in f
        assert "dsre_level" not in f
        assert not any(k.startswith("dsre_before_") for k in f.keys())
        assert not any(k.startswith("dsre_after_") for k in f.keys())


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
        _embed_output_metadata(path, pictures)
        result_pics = _extract_flac_pictures(path)
        assert len(result_pics) == 1
        assert result_pics[0].data == _FAKE_JXL
        assert result_pics[0].mime == "image/jxl"


def test_embed_output_metadata_no_crash_no_pictures():
    from DSRE import _embed_output_metadata
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        _embed_output_metadata(path, [])  # should not raise
