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


def test_scan_pending_excludes_output_subtree():
    '''OUTPUT_DIR が INPUT_DIR 配下の時、その全サブツリーを scan_pending が除外する。
    (実環境 C:\\Audio\\DSRE\\Output が C:\\Audio\\DSRE 配下にあり、処理済み出力を
    入力として拾うと全ライブラリを再編成してしまう回帰の防止)。'''
    from DSRE import WorkflowOrchestrator
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, 'Output')
        os.makedirs(out_dir)
        new1 = os.path.join(d, 'new1.flac')
        _mk_flac(new1, tags={'artist': 'X'})
        # 処理済み出力 (タグ無し) を OUTPUT_DIR に置く
        done = os.path.join(out_dir, 'done.flac')
        _mk_flac(done, freq=600.0, tags={'artist': 'Y'})
        orch = WorkflowOrchestrator(input_dir=d, output_dir=out_dir)
        pending = orch.scan_pending()
        assert new1 in pending
        assert done not in pending


def test_filter_candidates_drops_output_and_tagged():
    '''_filter_candidates は OUTPUT_DIR 配下と dsre_version タグ持ちを落とす。'''
    from DSRE import WorkflowOrchestrator
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, 'Output')
        os.makedirs(out_dir)
        keep = os.path.join(d, 'keep.flac')
        _mk_flac(keep, tags={'artist': 'X'})
        tagged = os.path.join(d, 'tagged.flac')
        _mk_flac(tagged, freq=500.0, tags={'artist': 'X', 'dsre_version': 'r99'})
        in_output = os.path.join(out_dir, 'out.flac')
        _mk_flac(in_output, freq=600.0)
        orch = WorkflowOrchestrator(input_dir=d, output_dir=out_dir)
        res = orch._filter_candidates([keep, tagged, in_output])
        assert res == [keep]


def test_resolve_multi_disc_album():
    '''album 内の disc 値で multi-disc 判定。'''
    from DSRE import WorkflowOrchestrator
    m_single1 = {'album': 'A', 'discnumber': '1'}
    m_single2 = {'album': 'A', 'discnumber': '1'}
    assert WorkflowOrchestrator._is_multi_disc([m_single1, m_single2], 'A') is False

    m_multi1 = {'album': 'A', 'discnumber': '1'}
    m_multi2 = {'album': 'A', 'discnumber': '2'}
    assert WorkflowOrchestrator._is_multi_disc([m_multi1, m_multi2], 'A') is True


def test_full_workflow_multi_version():
    """同曲 3 版 (1 版だけ tagged) でメタデータ伝播 + 最良選択 + 整列を確認。"""
    import shutil
    from DSRE import WorkflowOrchestrator, MetadataExtractor, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return

    # fpcalc は 96kHz 合成正弦波を処理できない (Empty fingerprint)。
    # 44100Hz / 5 秒の FLAC を生成するローカルヘルパーを使う。
    _SR44 = 44100

    def _mk44(path, freq=440.0, dur=5.0):
        t = np.linspace(0, dur, int(_SR44 * dur), endpoint=False)
        audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
        sf.write(path, audio, _SR44, subtype='PCM_24', format='FLAC')

    # ignore_cleanup_errors=True: SQLite WAL ファイルが残っても cleanup を通す。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        in_dir = os.path.join(d, "in")
        out_dir = os.path.join(d, "out")
        db = os.path.join(d, "fp.db")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # 同一音響コンテンツの 3 file を生成し、1 つだけタグを付与する。
        # shutil.copy で同一バイナリを複製することで content_hash が一致し、
        # fpcalc は 1 回の計算で 3 ファイルを同一 fingerprint として扱う。
        base = os.path.join(in_dir, "base_src.flac")
        _mk44(base)
        tagged = os.path.join(in_dir, "tagged.flac")
        u1 = os.path.join(in_dir, "untagged1.flac")
        u2 = os.path.join(in_dir, "untagged2.flac")
        shutil.copy(base, u1)
        shutil.copy(base, u2)
        shutil.copy(base, tagged)
        os.remove(base)

        from mutagen.flac import FLAC
        f = FLAC(tagged)
        for k, v in {"artist": "X", "album": "Y", "title": "Z",
                     "discnumber": "1", "tracknumber": "5",
                     "genre": "J-Pop"}.items():
            f[k] = [v]
        f.save()

        orch = WorkflowOrchestrator(input_dir=in_dir, output_dir=out_dir, db_path=db)
        bests = orch.run_stage1()
        del orch  # SQLite connection を明示的に解放

        # 1 個だけ残っている (3 -> 1)
        assert len(bests) == 1, f"expected 1 best, got {len(bests)}: {bests}"
        # 残った file には伝播されたタグがある
        m = MetadataExtractor.extract(bests[0])
        assert m["artist"] == "X"
        assert m["album"] == "Y"
        assert m["title"] == "Z"
        # foobar 階層に配置されている (genre/album が path に含まれる)
        assert "J-Pop" in bests[0], f"J-Pop not in {bests[0]}"
        assert "Y" in bests[0], f"Y not in {bests[0]}"
