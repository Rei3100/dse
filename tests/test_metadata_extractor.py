import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac_with_tags(path, tags: dict):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    f.save()


def test_extract_main_fields():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {
            "artist": "X", "album": "Y", "title": "Z",
            "discnumber": "2", "tracknumber": "7", "date": "2020",
            "genre": "J-Pop",
        })
        m = MetadataExtractor.extract(path)
        assert m["artist"] == "X"
        assert m["album"] == "Y"
        assert m["title"] == "Z"
        assert m["discnumber"] == "2"
        assert m["tracknumber"] == "7"
        assert m["date"] == "2020"
        assert m["genre"] == "J-Pop"


def test_extract_missing_fields_returns_empty():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {})
        m = MetadataExtractor.extract(path)
        assert m["artist"] == ""
        assert m["title"] == ""
        assert m["genre"] == ""


def test_extract_extended_tags():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {
            "circle": "C", "brand": "B", "version_info": "Live",
        })
        m = MetadataExtractor.extract(path)
        assert m["circle"] == "C"
        assert m["brand"] == "B"
        assert m["version_info"] == "Live"
