import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _chroma(seed):
    rng = np.random.RandomState(seed)
    return rng.rand(12, 256).astype(np.float32)


def test_registry_record_find_match():
    import DSRE
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = os.path.join(d, "r.db")
        outp = os.path.join(d, "song.flac")
        open(outp, "wb").write(b"x")  # 出力存在が条件
        reg = DSRE.ProcessedRegistry(db)
        ch = _chroma(1)
        reg.record(ch, 7.5, outp, '{"artist":["A"]}')
        m = reg.find_match(ch, 0.42)
        assert m is not None
        assert abs(m[1] - 7.5) < 1e-6 and m[2] == outp


def test_registry_no_match_below_threshold():
    import DSRE
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = os.path.join(d, "r.db")
        outp = os.path.join(d, "s.flac"); open(outp, "wb").write(b"x")
        reg = DSRE.ProcessedRegistry(db)
        reg.record(_chroma(1), 5.0, outp, "{}")
        # 無関係 chroma は高閾値で一致しない
        assert reg.find_match(_chroma(999), 0.9) is None


def test_registry_selfheal_missing_output():
    import DSRE
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db = os.path.join(d, "r.db")
        outp = os.path.join(d, "gone.flac"); open(outp, "wb").write(b"x")
        reg = DSRE.ProcessedRegistry(db)
        ch = _chroma(2)
        reg.record(ch, 5.0, outp, "{}")
        os.remove(outp)  # 出力消失
        assert reg.find_match(ch, 0.42) is None  # self-heal で無効化
        # エントリも除去されている
        import sqlite3
        with sqlite3.connect(db) as c:
            n = c.execute("SELECT COUNT(*) FROM processed_songs").fetchone()[0]
        assert n == 0
