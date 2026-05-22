import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
SR = 96000


def _mk(path, tags=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.linspace(0, 1.0, SR, endpoint=False)
    sf.write(path, (np.sin(2*np.pi*440*t)*0.3).astype(np.float32), SR,
             subtype="PCM_24", format="FLAC")
    if tags:
        from mutagen.flac import FLAC
        f = FLAC(path)
        for k, v in tags.items():
            f[k] = [v]
        f.save()


def test_scan_finds_subfolder_excludes_output_ref_processed():
    from DSRE import _scan_pending_files
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "Output")
        top = os.path.join(d, "top.flac"); _mk(top)
        sub = os.path.join(d, "Artist", "Album", "track.flac"); _mk(sub)  # relocated pending
        reff = os.path.join(d, "ref", "reference.flac"); _mk(reff)       # reference -> excluded
        outf = os.path.join(out, "done.flac"); _mk(outf)                 # output -> excluded
        proc = os.path.join(d, "Proc", "p.flac"); _mk(proc, {"dsre_version": "r1"})  # processed -> excluded
        res = set(os.path.normcase(p) for p in _scan_pending_files(d, out))
        assert os.path.normcase(top) in res
        assert os.path.normcase(sub) in res        # サブフォルダの未処理 file を拾う
        assert os.path.normcase(reff) not in res   # ref 除外
        assert os.path.normcase(outf) not in res   # OUTPUT 除外
        assert os.path.normcase(proc) not in res   # dsre_version 済 除外


def test_sort_output_into_foobar(monkeypatch):
    import DSRE
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "Output"); os.makedirs(out)
        monkeypatch.setattr(DSRE, "OUTPUT_DIR", out)
        f = os.path.join(out, "flat.flac")
        _mk(f, {"artist": "A", "album": "AL", "title": "T",
                "genre": "J-Pop", "tracknumber": "3", "discnumber": "1"})
        newp = DSRE._sort_output_into_foobar(f)
        assert os.path.exists(newp)
        assert newp != f                     # 移動された
        assert "J-Pop" in newp and "AL" in newp  # foobar 階層
        assert not os.path.exists(f)
