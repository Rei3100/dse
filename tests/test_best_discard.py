import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path, dur=1.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def test_select_best_clean():
    from DSRE import BestSelector, ScoreResult
    a = ScoreResult(80.0, {}, False, [])
    b = ScoreResult(70.0, {}, False, [])
    c = ScoreResult(90.0, {}, True, ["lossy_hf_cliff"])  # flagged
    items = [("a.flac", a, 1000), ("b.flac", b, 2000), ("c.flac", c, 3000)]
    best = BestSelector.choose(items)
    assert best.path == "a.flac"
    assert best.warn_all_flagged is False


def test_select_best_all_flagged():
    from DSRE import BestSelector, ScoreResult
    a = ScoreResult(60.0, {}, True, ["brick_wall_flatness"])
    b = ScoreResult(75.0, {}, True, ["hyper_compression"])
    items = [("a", a, 1000), ("b", b, 1500)]
    best = BestSelector.choose(items)
    assert best.path == "b"
    assert best.warn_all_flagged is True


def test_select_tiebreak_filesize():
    from DSRE import BestSelector, ScoreResult
    a = ScoreResult(80.0, {}, False, [])
    b = ScoreResult(80.0, {}, False, [])
    items = [("a", a, 1000), ("b", b, 2000)]
    best = BestSelector.choose(items)
    assert best.path == "b"


def test_discard_rename_and_trash():
    from DSRE import DiscardHandler
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "loser.flac")
        _mk_flac(f)
        DiscardHandler.discard(f, dry_run=True)
        renamed = os.path.join(d, "[DSRE-inferior] loser.flac")
        assert os.path.isfile(renamed)
        assert not os.path.isfile(f)
