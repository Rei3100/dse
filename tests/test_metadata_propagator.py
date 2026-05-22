import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _mk(path, tags, with_artwork=False, art_mime="image/jxl"):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC, Picture
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    if with_artwork:
        pic = Picture(); pic.type = 3; pic.mime = art_mime; pic.data = _FAKE_JXL
        f.add_picture(pic)
    f.save()


def test_jxl_artwork_given_to_artless_canonical():
    """canonical (=最良候補) が jxl を持たず、別 file に jxl がある場合、
    canonical にも jxl が付与される (最良がアートワーク無しを防ぐ)。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")  # 完備メタ・アートワーク無し → canonical
        _mk(a, {"artist": "X", "album": "AL", "title": "T", "genre": "G"})
        b = os.path.join(d, "b.flac")  # 低メタ・jxl 有り
        _mk(b, {"artist": "Y"}, with_artwork=True, art_mime="image/jxl")
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        assert MetadataPropagator.choose_canonical(meta)["__path__"] == a
        MetadataPropagator.propagate(meta)
        af = FLAC(a)
        assert len(af.pictures) == 1
        assert af.pictures[0].mime == "image/jxl"


def test_non_jxl_artwork_not_unified():
    """jxl 以外のアートワークは統合しない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")  # 完備メタ・png アートワーク → canonical
        _mk(a, {"artist": "X", "album": "AL", "title": "T", "genre": "G"},
            with_artwork=True, art_mime="image/png")
        b = os.path.join(d, "b.flac")  # 低メタ・アートワーク無し
        _mk(b, {"artist": "Y"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert len(bf.pictures) == 0  # png は伝播されない


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


def test_propagate_unifies_conflicting_value():
    """非 version タグは canonical 値で上書き統一される (差を残さない)。
    canonical は完備度最大 file なので良メタが潰される事故は起きない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "album": "AL", "title": "T", "genre": "G"})
        b = os.path.join(d, "b.flac")
        _mk(b, {"genre": "B-genre"})  # b は低メタ (canonical にならない)
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        canon = MetadataPropagator.choose_canonical(meta)
        assert canon["__path__"] == a  # 完備な a が canonical
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["genre"][0] == "G"      # B-genre は canonical G で統一
        assert bf["artist"][0] == "X"
        assert bf["album"][0] == "AL"
        assert bf["title"][0] == "T"


def test_unify_covers_arbitrary_and_future_tags():
    """固定リストに無いタグ (composer 等) も統一される。per-file 技術タグ
    (replaygain_*) は各音源固有なので統一しない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "album": "AL", "title": "T",
                "composer": "C", "lyricist": "LY", "customtag": "CUSTOM",
                "replaygain_track_range": "2.68 LU"})
        b = os.path.join(d, "b.flac")
        _mk(b, {"artist": "Y", "description": "b-only",
                "replaygain_track_range": "2.67 LU"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        assert MetadataPropagator.choose_canonical(meta)["__path__"] == a
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        # 固定リスト外タグも統一
        assert bf["composer"][0] == "C"
        assert bf["lyricist"][0] == "LY"
        assert bf["customtag"][0] == "CUSTOM"
        assert bf["artist"][0] == "X"
        # canonical に無い target 固有 identity タグは削除 (差を消す)
        assert "description" not in bf
        # per-file 技術タグは b 自身の値を保持 (統一しない)
        assert bf["replaygain_track_range"][0] == "2.67 LU"


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
