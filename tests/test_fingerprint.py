import os, sys, tempfile, sqlite3
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path: str):
    """chromaprint が処理できる 5 秒・多倍音 FLAC を生成。"""
    t = np.linspace(0, 5.0, SR * 5, endpoint=False)
    audio = np.zeros(SR * 5, dtype=np.float32)
    for freq in [220, 440, 880, 1760, 3520]:
        audio += np.sin(2 * np.pi * freq * t) * 0.06
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def test_compute_returns_duration_and_fp():
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        flac = os.path.join(d, "test.flac")
        _mk_flac(flac)
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        result = eng.compute(flac)
        assert result is not None
        assert result.duration_sec > 1.5
        assert isinstance(result.fingerprint, str)
        assert len(result.fingerprint) > 0


def test_cache_hit():
    """2 回目は cache 経由 (subprocess を呼ばずに同結果)。"""
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        flac = os.path.join(d, "test.flac")
        _mk_flac(flac)
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        r1 = eng.compute(flac)
        r2 = eng.compute(flac)
        assert r1.fingerprint == r2.fingerprint
        # cache table 確認
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT * FROM fingerprints").fetchall()
            assert len(rows) == 1


def test_compute_failure_returns_none():
    """壊れた file は None を返す。"""
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        broken = os.path.join(d, "broken.flac")
        with open(broken, "wb") as f:
            f.write(b"not a flac file")
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        result = eng.compute(broken)
        assert result is None
