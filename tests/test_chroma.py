import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _seq(pattern, n=256):
    """12 x n chroma: pattern は各時間ブロックで強い pitch class の列。"""
    out = np.zeros((12, n), dtype=np.float32)
    blocks = np.array_split(np.arange(n), len(pattern))
    for blk, pc in zip(blocks, pattern):
        out[pc % 12, blk] = 1.0
    return out


def test_chroma_identical_is_one():
    from DSRE import chroma_similarity
    a = _seq([0, 4, 7, 5, 2, 9, 11, 0])
    assert chroma_similarity(a, a) > 0.999


def test_chroma_transposition_invariant():
    """全体を半音シフト (移調) しても同曲と判定する。"""
    from DSRE import chroma_similarity
    a = _seq([0, 4, 7, 5, 2, 9, 11, 0])
    b = _seq([(p + 3) for p in [0, 4, 7, 5, 2, 9, 11, 0]])  # +3 半音
    assert chroma_similarity(a, b) > 0.95


def test_chroma_different_song_is_low():
    """無関係な進行は低スコア。"""
    from DSRE import chroma_similarity
    a = _seq([0, 4, 7, 5, 2, 9, 11, 0])
    b = _seq([1, 6, 3, 10, 8, 1, 6, 3])
    assert chroma_similarity(a, b) < 0.6


def test_chroma_small_offset_tolerated():
    """開始位置が数フレームずれても同曲と判定する (時間ラグ探索)。"""
    from DSRE import chroma_similarity
    a = _seq([0, 4, 7, 5, 2, 9, 11, 0])
    b = np.roll(a, 5, axis=1)  # 5 フレーム遅延
    assert chroma_similarity(a, b) > 0.9
