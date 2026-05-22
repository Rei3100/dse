import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk(path, tags):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    f.save()


def test_split_same_version_one_subcluster():
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        subs = split_versions(meta)
        assert len(subs) == 1
        assert len(subs[0]) == 2


def test_split_different_version_two_subclusters():
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "version_info": "Live"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "version_info": "Studio"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        subs = split_versions(meta)
        assert len(subs) == 2
        for s in subs:
            assert len(s) == 1


def test_split_multiple_version_tags():
    """複数の version 系タグの組み合わせで判定。"""
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "live_type": "L"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "live_type": "L"})
        c = os.path.join(d, "c.flac"); _mk(c, {"artist": "X", "live_type": "M"})
        meta = [MetadataExtractor.extract(p) for p in (a, b, c)]
        subs = split_versions(meta)
        assert len(subs) == 2
        sizes = sorted(len(s) for s in subs)
        assert sizes == [1, 2]
