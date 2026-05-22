import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_prune_removes_empty_chain_up_to_boundary():
    import DSRE
    with tempfile.TemporaryDirectory() as d:
        boundary = os.path.join(d, "IN")
        deep = os.path.join(boundary, "Artist", "Album")
        os.makedirs(deep)
        DSRE._prune_empty_dirs(deep, boundary=boundary)
        assert not os.path.exists(os.path.join(boundary, "Artist"))
        assert os.path.exists(boundary)  # boundary 自身は残す


def test_prune_stops_at_nonempty():
    import DSRE
    with tempfile.TemporaryDirectory() as d:
        boundary = os.path.join(d, "IN")
        deep = os.path.join(boundary, "Artist", "Album")
        os.makedirs(deep)
        # Artist に別ファイルを置く → Album は消えるが Artist は残る
        with open(os.path.join(boundary, "Artist", "keep.txt"), "w") as f:
            f.write("x")
        DSRE._prune_empty_dirs(deep, boundary=boundary)
        assert not os.path.exists(deep)
        assert os.path.exists(os.path.join(boundary, "Artist"))


def test_prune_never_removes_output_dir(monkeypatch):
    import DSRE
    with tempfile.TemporaryDirectory() as d:
        boundary = os.path.join(d, "IN")
        out = os.path.join(boundary, "Output")
        os.makedirs(out)
        monkeypatch.setattr(DSRE, "OUTPUT_DIR", out)
        DSRE._prune_empty_dirs(out, boundary=boundary)
        assert os.path.exists(out)  # OUTPUT_DIR は空でも消さない
