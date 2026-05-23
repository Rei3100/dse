import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_save_flac24_out_long_filename():
    """多人数 artist 等で出力ファイル名が ~250 字級になっても保存が成功する。
    旧実装は tmp = final + '.tmp_src.flac' で 1 コンポーネント 255 字制限を超え
    sf.write が失敗していた (無限L∞PだLOVE♡ 全員版で実際に失敗)。短い uuid tmp で解決。
    ffmpeg 不在環境でもメタ無しフォールバックで出力されることを確認。"""
    import DSRE
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        # 入力 (短い名前)
        sr = 96000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        in_path = os.path.join(d, "in.flac")
        sf.write(in_path, audio, sr, subtype="PCM_24", format="FLAC")
        # 出力 (254 字級のファイル名 — 実際の全員版相当)
        outdir = os.path.join(d, "out")
        os.makedirs(outdir)
        long_name = ("01.001.無限L∞PだLOVE - " + "あ" * 228 + ".flac")
        assert 248 < len(long_name) < 255  # 単体は有効、tmp suffix を足すと 255 超
        out_path = os.path.join(outdir, long_name)
        y_out = audio.reshape(1, -1)  # (channels, samples) for ndim==2 path
        final = DSRE.save_flac24_out(in_path, y_out, sr, out_path)
        assert os.path.exists(DSRE._lp(final)), "long-name output not written"
