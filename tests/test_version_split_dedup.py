import os, sys, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
SR=96000
def _mk(path, tags):
    t=np.linspace(0,1.0,SR,endpoint=False)
    sf.write(path,(np.sin(2*np.pi*440*t)*0.3).astype(np.float32),SR,subtype="PCM_24",format="FLAC")
    from mutagen.flac import FLAC
    f=FLAC(path)
    for k,v in tags.items(): f[k]=[v]
    f.save()

def test_instrumental_splits_from_vocal():
    """vocal_type=Instrumental は別バージョンとして分割される。"""
    from DSRE import _split_versions_for_dedup, MetadataExtractor
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a=os.path.join(d,"v1.flac"); _mk(a,{"title":"S","artist":"X"})
        b=os.path.join(d,"v2.flac"); _mk(b,{"title":"S","artist":"X"})
        c=os.path.join(d,"inst.flac"); _mk(c,{"title":"S","artist":"X","vocal_type":"Instrumental"})
        meta=[MetadataExtractor.extract(p) for p in (a,b,c)]
        groups=_split_versions_for_dedup(meta)
        assert len(groups)==2  # vocal(2) + instrumental(1)
        sizes=sorted(len(g) for g in groups)
        assert sizes==[1,2]

def test_cover_type_does_not_split():
    """cover_type 違いは分割しない (同一視聴体験 → 1 個に dedup)。"""
    from DSRE import _split_versions_for_dedup, MetadataExtractor
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        a=os.path.join(d,"a.flac"); _mk(a,{"title":"S","artist":"X","cover_type":"Cover"})
        b=os.path.join(d,"b.flac"); _mk(b,{"title":"S","artist":"Y"})
        meta=[MetadataExtractor.extract(p) for p in (a,b)]
        groups=_split_versions_for_dedup(meta)
        assert len(groups)==1  # カバーは同一グループ
