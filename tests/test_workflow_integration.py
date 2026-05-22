import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path, freq=440.0, dur=2.0, tags=None):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype='PCM_24', format='FLAC')
    if tags:
        from mutagen.flac import FLAC
        f = FLAC(path)
        for k, v in tags.items():
            f[k] = [v]
        f.save()


def test_idempotency_skip_already_processed():
    '''dsre_version タグ持ち file は WorkflowOrchestrator.scan_pending で skip。'''
    from DSRE import WorkflowOrchestrator
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, 'a.flac')
        _mk_flac(a, tags={'artist': 'X', 'album': 'Y', 'title': 'Z', 'dsre_version': 'r99'})
        b = os.path.join(d, 'b.flac')
        _mk_flac(b, freq=600.0, tags={'artist': 'X'})
        orch = WorkflowOrchestrator(input_dir=d, output_dir=d)
        pending = orch.scan_pending()
        assert b in pending
        assert a not in pending


def test_resolve_multi_disc_album():
    '''album 内の disc 値で multi-disc 判定。'''
    from DSRE import WorkflowOrchestrator
    m_single1 = {'album': 'A', 'discnumber': '1'}
    m_single2 = {'album': 'A', 'discnumber': '1'}
    assert WorkflowOrchestrator._is_multi_disc([m_single1, m_single2], 'A') is False

    m_multi1 = {'album': 'A', 'discnumber': '1'}
    m_multi2 = {'album': 'A', 'discnumber': '2'}
    assert WorkflowOrchestrator._is_multi_disc([m_multi1, m_multi2], 'A') is True
