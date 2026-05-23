import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _meta(**kw):
    base = {k: "" for k in [
        "artist", "album", "title", "discnumber", "tracknumber",
        "date", "genre", "age", "circle", "category", "source", "grouping",
        "franchises", "products", "series", "brand", "subtitle", "elements",
        "project", "collaboration", "group", "unit", "album_type",
        "featuring", "produced", "arrange_type",
        "version_info", "remaster_info", "cover_type", "live_type",
        "vocal_type", "m_number",
    ]}
    base.update(kw)
    return base


def test_full_metadata_path():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="Artist", album="Album", title="Title",
        discnumber="1", tracknumber="3", date="2020",
        genre="J-Pop", circle="Circle",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path == r"C:/out\J-Pop\Circle\Album (2020)\01.003.Title - Artist.flac"


def test_missing_levels_omitted():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="T", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path == r"C:/out\A\01.001.T - X.flac"


def test_multi_disc_layer():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="T", discnumber="2", tracknumber="5")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=True)
    assert path == r"C:/out\A\Disc 02\02.005.T - X.flac"


def test_sanitize_forbidden_chars():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="Hi? Yes!", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert "?" not in path
    assert "？" in path


def test_parent_child_dedup_products_eq_brand():
    """products == brand なら products 階層は省略。"""
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        brand="SameBrand", products="SameBrand",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.count("SameBrand") == 1


def test_modifier_suffix_appended():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        version_info="Live", cover_type="Self",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.endswith(" [Live] [Self].flac")


def test_featuring_in_artist_part():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        featuring="Y",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.endswith("T - X [feat. Y].flac")


def test_long_path_raw_then_lp_prefix():
    r"""build は生パスを返し (prefix なし)、_lp() が正しい 4 文字 \\?\ を付ける。
    旧実装は build 内で 3 文字 \?\ を付け rename/makedirs を壊していた。"""
    from DSRE import FoobarPathBuilder, _lp
    import os
    long = "Long" * 80
    m = _meta(artist="X", album=long, title="T", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/very/long/output", m, multi_disc=False)
    assert not path.startswith("\\\\?\\")  # build は生パス
    if os.name == "nt":
        assert _lp(path).startswith("\\\\?\\")  # _lp が正しい 4 文字 prefix
