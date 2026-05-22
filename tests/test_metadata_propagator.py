import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _mk(path, tags, with_artwork=False):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC, Picture
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    if with_artwork:
        pic = Picture(); pic.type = 3; pic.mime = "image/jxl"; pic.data = _FAKE_JXL
        f.add_picture(pic)
    f.save()


def test_canonical_selection_by_tag_richness():
    from DSRE import MetadataPropagator, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "album": "Y", "title": "Z", "date": "2020"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X"})
        c = os.path.join(d, "c.flac"); _mk(c, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b, c)]
        canon = MetadataPropagator.choose_canonical(meta)
        assert canon["__path__"] == a


def test_canonical_artwork_bonus():
    from DSRE import MetadataPropagator, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "album": "Y"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "album": "Y"}, with_artwork=True)
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        canon = MetadataPropagator.choose_canonical(meta)
        assert canon["__path__"] == b


def test_propagate_to_untagged():
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "album": "Y", "title": "Z", "genre": "G"}, with_artwork=True)
        b = os.path.join(d, "b.flac")
        _mk(b, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["artist"][0] == "X"
        assert bf["album"][0] == "Y"
        assert bf["title"][0] == "Z"
        assert bf["genre"][0] == "G"
        assert len(bf.pictures) == 1
        assert bf.pictures[0].mime == "image/jxl"


def test_propagate_respects_existing_value():
    """伝播先が既値を持つタグは上書きしない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "genre": "G"})
        b = os.path.join(d, "b.flac")
        _mk(b, {"genre": "B-genre"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["genre"][0] == "B-genre"
        assert bf["artist"][0] == "X"


def test_version_tags_never_propagated():
    """version 識別系タグは canonical が持っていても伝播しない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "version_info": "Live", "live_type": "L"})
        b = os.path.join(d, "b.flac")
        _mk(b, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["artist"][0] == "X"
        assert "version_info" not in bf
        assert "live_type" not in bf
