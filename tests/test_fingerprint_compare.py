import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# fpcalc は 96kHz を扱えない (Empty fingerprint) ため 44100Hz で生成する
SR = 44100


def _mk_flac(path, freq=440.0, dur=3.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_16", format="FLAC")


def test_same_audio_similarity_high():
    from DSRE import FingerprintEngine, fingerprint_similarity, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a = os.path.join(d, "a.flac")
        b = os.path.join(d, "b.flac")
        _mk_flac(a, 440.0)
        _mk_flac(b, 440.0)  # same content, different file
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        fa = eng.compute(a)
        fb = eng.compute(b)
        assert fa is not None and fb is not None, "fpcalc failed to compute fingerprint"
        sim = fingerprint_similarity(fa.fingerprint, fb.fingerprint)
        assert sim > 0.95


def test_different_audio_similarity_low():
    from DSRE import FingerprintEngine, fingerprint_similarity, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a = os.path.join(d, "a.flac")
        b = os.path.join(d, "b.flac")
        _mk_flac(a, 440.0, dur=3.0)
        _mk_flac(b, 800.0, dur=3.0)  # different frequency = different audio
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        fa = eng.compute(a)
        fb = eng.compute(b)
        assert fa is not None and fb is not None, "fpcalc failed to compute fingerprint"
        sim = fingerprint_similarity(fa.fingerprint, fb.fingerprint)
        # 純音同士は実音楽より高め: 0.75 以下であれば十分に「異なる」と判定できる
        assert sim < 0.75
