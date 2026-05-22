import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_cluster_single_pair():
    from DSRE import ClusterBuilder
    cb = ClusterBuilder(similarity_threshold=0.85)
    cb.add_pair("a.flac", "b.flac", 0.92)
    clusters = cb.build(items=["a.flac", "b.flac", "c.flac"])
    # a-b 連結、c は単独
    assert len(clusters) == 2
    cluster_with_a = next(c for c in clusters if "a.flac" in c)
    assert "b.flac" in cluster_with_a
    assert ["c.flac"] in clusters


def test_cluster_transitive():
    """a-b と b-c の連結で a-b-c が 1 cluster。"""
    from DSRE import ClusterBuilder
    cb = ClusterBuilder(similarity_threshold=0.85)
    cb.add_pair("a", "b", 0.90)
    cb.add_pair("b", "c", 0.91)
    clusters = cb.build(items=["a", "b", "c"])
    assert len(clusters) == 1
    assert set(clusters[0]) == {"a", "b", "c"}


def test_cluster_threshold():
    """閾値未満のペアは連結しない。"""
    from DSRE import ClusterBuilder
    cb = ClusterBuilder(similarity_threshold=0.85)
    cb.add_pair("a", "b", 0.80)  # 閾値未満
    clusters = cb.build(items=["a", "b"])
    assert len(clusters) == 2


def test_cluster_all_singleton():
    """ペア未登録の場合、全 item が単独 cluster。"""
    from DSRE import ClusterBuilder
    cb = ClusterBuilder(similarity_threshold=0.85)
    clusters = cb.build(items=["a", "b", "c"])
    assert len(clusters) == 3
