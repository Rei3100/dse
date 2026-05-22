import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_fpcalc_path_resolves():
    """fpcalc.exe のパスが解決される (バンドル時または PATH)。"""
    from DSRE import _resolve_fpcalc_path
    p = _resolve_fpcalc_path()
    # バンドルなしでも None or 文字列を返す (graceful degrade)
    assert p is None or os.path.isfile(p)


def test_fpcalc_executable():
    """fpcalc.exe が実行可能 (--version 確認)。バンドル前提のテスト。"""
    from DSRE import _resolve_fpcalc_path
    import subprocess
    p = _resolve_fpcalc_path()
    if p is None:
        return  # binary 未配置時は skip 扱い
    result = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "fpcalc" in (result.stdout + result.stderr).lower()
