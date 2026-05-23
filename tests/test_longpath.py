import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_lp_short_path_unchanged():
    import DSRE
    p = r"C:\Audio\DSRE\short.flac"
    assert DSRE._lp(p) == p


def test_lp_long_path_prefixed_on_windows():
    import DSRE
    long_name = "x" * 300 + ".flac"
    p = os.path.join(r"C:\Audio\DSRE", long_name)
    out = DSRE._lp(p)
    if os.name == "nt":
        assert out.startswith("\\\\?\\")          # 拡張プレフィックス付与
        assert out.endswith(long_name)
    else:
        assert out == p                            # 非 Windows は無変換


def test_lp_already_prefixed_unchanged():
    import DSRE
    p = "\\\\?\\C:\\Audio\\DSRE\\" + "y" * 300 + ".flac"
    assert DSRE._lp(p) == p


def test_lp_empty_and_none_safe():
    import DSRE
    assert DSRE._lp("") == ""
