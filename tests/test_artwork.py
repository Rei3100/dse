import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"  # JXL signature bytes (最小フェイク)


def _make_test_flac(path: str) -> None:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3] * 2, axis=1).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def _embed_picture(flac_path: str, img_bytes: bytes, mime: str = "image/jxl") -> None:
    from mutagen.flac import FLAC, Picture
    f = FLAC(flac_path)
    pic = Picture()
    pic.type = 3  # Front cover
    pic.mime = mime
    pic.data = img_bytes
    f.add_picture(pic)
    f.save()


def test_extract_flac_pictures_returns_list():
    from DSRE import _extract_flac_pictures
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.flac")
        _make_test_flac(path)
        result = _extract_flac_pictures(path)
        assert isinstance(result, list)
        assert len(result) == 0  # アートワーク無しは空リスト


def test_extract_flac_pictures_captures_jxl():
    from DSRE import _extract_flac_pictures
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.flac")
        _make_test_flac(path)
        _embed_picture(path, _FAKE_JXL, mime="image/jxl")
        result = _extract_flac_pictures(path)
        assert len(result) == 1
        assert result[0].data == _FAKE_JXL
        assert result[0].mime == "image/jxl"


def test_extract_flac_pictures_nonexistent_returns_empty():
    from DSRE import _extract_flac_pictures
    result = _extract_flac_pictures("/nonexistent/path.flac")
    assert result == []
