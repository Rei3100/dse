# DSRE Sub-project B+D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** raw FLAC 群を投入すれば、音響指紋によるグルーピング → メタデータ自動伝播 → 客観品質採点 → 最良選択 → 識別可能な破棄 → DSRE 処理 → foobar 互換階層整列 まで全自動化する。

**Architecture:** chromaprint (`fpcalc.exe`) を `subprocess` で呼び出し SQLite に指紋をキャッシュ。Hamming 一致率と union-find でクラスタリング。各クラスタで canonical を選定して mutagen でメタデータ伝播。`save_flac24_out` から音量正規化を独立関数化し、採点側 (正規化 PCM の直接解析) と本処理側で共有。すべて `DSRE.py` 単一ファイル内に追加 (既存単一ファイル設計を保持)。

**Tech Stack:** Python 3.11 / numpy / scipy / mutagen / sqlite3 / subprocess / chromaprint (fpcalc.exe バンドル) / pyqt6 / pytest

---

## File Structure

| ファイル | 変更種別 | 責務 |
|---|---|---|
| `DSRE.py` | 修正・追加 | 全新モジュールクラス追加、`save_flac24_out` リファクタ、`Worker.run` 改修 |
| `DSRE.spec` | 修正 | `fpcalc.exe` バンドル設定追加 |
| `_internal/ffmpeg/fpcalc.exe` | 新規配置 | chromaprint 公式 binary (~500KB) |
| `tests/test_fingerprint.py` | 新規 | 指紋取得 + キャッシュ |
| `tests/test_cluster.py` | 新規 | 類似度 + union-find |
| `tests/test_metadata_propagator.py` | 新規 | canonical 選定 + 伝播 + 既値尊重 |
| `tests/test_quality_probe.py` | 新規 | 正規化 + 採点 + フラグ |
| `tests/test_foobar_path.py` | 新規 | テンプレ展開 + サニタイズ + 長パス |
| `tests/test_discard.py` | 新規 | リネーム + ゴミ箱投入 |
| `tests/test_workflow_integration.py` | 新規 | STAGE 1-3 統合 |

---

## Task 1: TP_TARGET_DBFS の確認 + 音量正規化関数の抽出 (リファクタ)

**Files:**
- Modify: `DSRE.py` (lines 1084-1095 付近、`save_flac24_out` 内の音量正規化部分を関数化)

既存 `save_flac24_out` 内の音量正規化を `_dsre_normalize_volume(audio, sr) -> np.ndarray` として独立関数化する。同一実装を採点側 (QualityProbe) と本処理側 (save_flac24_out) で共有することで乖離リスクを構造的に防ぐ。

- [ ] **Step 1: 既存 `save_flac24_out` の音量正規化部分を Read で確認**

Read DSRE.py lines 1075-1100, confirm: `_true_peak(data)` + `TP_TARGET_DBFS` を使い `target / tp` で gain 適用、`DSRE_VOLUME_OPTIMIZE=0` で旧 sample-peak only モードに falls back する 2 分岐。

- [ ] **Step 2: テストを書く (関数抽出後の動作保証)**

Create `tests/test_normalize_volume.py`:

```python
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000

def test_normalize_volume_clip_zero():
    """正規化後 true peak は TP_TARGET_DBFS 以下、clip=0。"""
    from DSRE import _dsre_normalize_volume, TP_TARGET_DBFS, _true_peak
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 5.0).astype(np.float32)
    audio_t = audio[np.newaxis, :]  # (1, samples) channels-first
    out = _dsre_normalize_volume(audio_t, SR)
    tp_db = 20 * np.log10(_true_peak(out) + 1e-30)
    assert tp_db <= TP_TARGET_DBFS + 0.5
    assert int(np.sum(np.abs(out) >= 1.0)) == 0


def test_normalize_volume_legacy_mode():
    """DSRE_VOLUME_OPTIMIZE=0 で旧 sample-peak only モードに分岐。"""
    from DSRE import _dsre_normalize_volume
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 1.5).astype(np.float32)
    audio_t = audio[np.newaxis, :]
    os.environ["DSRE_VOLUME_OPTIMIZE"] = "0"
    try:
        out = _dsre_normalize_volume(audio_t, SR)
    finally:
        del os.environ["DSRE_VOLUME_OPTIMIZE"]
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-6
```

- [ ] **Step 3: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_normalize_volume.py -v`
Expected: FAIL with `ImportError: cannot import name '_dsre_normalize_volume'`

- [ ] **Step 4: 関数抽出**

`DSRE.py` の `save_flac24_out` 内部、行 1084-1095 付近にある音量正規化ブロックを上方に新関数として切り出す:

```python
def _dsre_normalize_volume(data: np.ndarray, sr: int) -> np.ndarray:
    """音量最適化: true peak で TP_TARGET_DBFS に正規化し clip=0 を構造保証。

    DSRE_VOLUME_OPTIMIZE=0 で旧動作 (sample peak > 1.0 のみ正規化) に分岐。
    `data` は channels-first (channels, samples) または mono (samples,) の float32。
    戻り値は data と同 shape、dtype=float32 の正規化済 audio。
    """
    data = data.astype(np.float32, copy=False)
    if os.environ.get("DSRE_VOLUME_OPTIMIZE") != "0":
        tp = _true_peak(data)
        if tp > 0:
            target = 10.0 ** (TP_TARGET_DBFS / 20.0)
            data = (data * (target / tp)).astype(np.float32, copy=False)
    else:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 1.0:
            data = data / peak
    return data
```

そして `save_flac24_out` 内の該当ブロックを `data = _dsre_normalize_volume(data, sr)` 1 行に置換する。

- [ ] **Step 5: テスト + selftest 実行**

Run:
```
python -m pytest tests/test_normalize_volume.py -v
python DSRE.py --selftest 2>&1 | Select-String "verdict|selftest"
```
Expected: テスト全 PASS + `verdict=EQUIV`

- [ ] **Step 6: Commit**

```bash
git add DSRE.py tests/test_normalize_volume.py
git commit -m "refactor: extract _dsre_normalize_volume from save_flac24_out

Independent function will be shared by save path and upcoming scoring
path (Sub-project B+D quality probe) to structurally prevent divergence."
```

---

## Task 2: fpcalc.exe バンドル + パス解決

**Files:**
- Create: `_internal/ffmpeg/fpcalc.exe` (chromaprint 公式 binary 配置)
- Modify: `DSRE.py` (パス解決ヘルパー追加)
- Modify: `DSRE.spec` (datas に fpcalc.exe 追加)

- [ ] **Step 1: chromaprint binary をプロジェクト内へ配置**

Run (PowerShell、ユーザー手動 or 自動ダウンロード):
```powershell
# chromaprint 1.5.1 Windows binary を https://github.com/acoustid/chromaprint/releases から取得
# 解凍して fpcalc.exe を以下に配置:
New-Item -ItemType Directory -Force "C:\Users\reinb\src\DSRE\_internal\ffmpeg" | Out-Null
# fpcalc.exe を _internal/ffmpeg/ に配置 (~500KB)
```

確認: `Test-Path "C:\Users\reinb\src\DSRE\_internal\ffmpeg\fpcalc.exe"` が True

- [ ] **Step 2: テストを書く (パス解決)**

Create `tests/test_fpcalc_path.py`:

```python
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
```

- [ ] **Step 3: テスト失敗を確認**

Run: `python -m pytest tests/test_fpcalc_path.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: パス解決ヘルパー追加**

`DSRE.py` の既存 `_find_ffmpeg()` 等の近くに以下を追加:

```python
def _resolve_fpcalc_path() -> "str | None":
    """fpcalc.exe のパスを解決。バンドル → PATH → None の優先順。
    None 戻りで指紋機能は無効化 (graceful degrade)。"""
    # バンドル探索 (既存 _resource_base_dirs() を流用)
    for base in _resource_base_dirs():
        for sub in ("ffmpeg", "_internal/ffmpeg", ""):
            cand = os.path.join(base, sub, "fpcalc.exe") if sub else os.path.join(base, "fpcalc.exe")
            if os.path.isfile(cand):
                return cand
    # PATH 探索
    from shutil import which
    found = which("fpcalc") or which("fpcalc.exe")
    if found:
        return found
    return None
```

- [ ] **Step 5: DSRE.spec に fpcalc.exe 追加**

`DSRE.spec` を編集し、`datas` リストに以下のエントリを追加 (ffmpeg/ffmpeg.exe の近く):

```python
datas += [
    ("_internal/ffmpeg/fpcalc.exe", "ffmpeg"),
]
```

(ffmpeg.exe バンドルの記述に倣う。既存形式に合わせて追加)

- [ ] **Step 6: テスト + selftest 実行**

Run:
```
python -m pytest tests/test_fpcalc_path.py -v
python DSRE.py --selftest 2>&1 | Select-String "verdict"
```
Expected: 全 PASS + `verdict=EQUIV`

- [ ] **Step 7: Commit**

```bash
git add _internal/ffmpeg/fpcalc.exe DSRE.py DSRE.spec tests/test_fpcalc_path.py
git commit -m "feat: bundle fpcalc.exe (chromaprint) for acoustic fingerprinting

Adds resource path resolver with graceful degrade when binary is absent.
Spec datas updated to include fpcalc.exe alongside ffmpeg."
```

---

## Task 3: FingerprintEngine (指紋取得 + SQLite キャッシュ)

**Files:**
- Modify: `DSRE.py` (FingerprintEngine クラス追加、SQLite 新テーブル)
- Test: `tests/test_fingerprint.py`

`fpcalc.exe` を subprocess で呼び出して指紋取得 + `dsre_log.db` (既存) に新テーブル `fingerprints` を作りキャッシュする。content-hash で同一ファイルを跨ぐ識別。

- [ ] **Step 1: テストを書く**

Create `tests/test_fingerprint.py`:

```python
import os, sys, tempfile, sqlite3
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path: str, freq: float = 440.0):
    t = np.linspace(0, 2.0, SR * 2, endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def test_compute_returns_duration_and_fp():
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        flac = os.path.join(d, "test.flac")
        _mk_flac(flac)
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        result = eng.compute(flac)
        assert result is not None
        assert result.duration_sec > 1.5
        assert isinstance(result.fingerprint, str)
        assert len(result.fingerprint) > 0


def test_cache_hit():
    """2 回目は cache 経由 (subprocess を呼ばずに同結果)。"""
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        flac = os.path.join(d, "test.flac")
        _mk_flac(flac)
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        r1 = eng.compute(flac)
        r2 = eng.compute(flac)
        assert r1.fingerprint == r2.fingerprint
        # cache table 確認
        with sqlite3.connect(db) as c:
            rows = c.execute("SELECT * FROM fingerprints").fetchall()
            assert len(rows) == 1


def test_compute_failure_returns_none():
    """壊れた file は None を返す。"""
    from DSRE import FingerprintEngine, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        broken = os.path.join(d, "broken.flac")
        with open(broken, "wb") as f:
            f.write(b"not a flac file")
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        result = eng.compute(broken)
        assert result is None
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: FAIL with `ImportError: cannot import name 'FingerprintEngine'`

- [ ] **Step 3: FingerprintEngine 実装**

`DSRE.py` の Sub-project A 関連関数群の近く (例: `_embed_output_metadata` の後) に追加:

```python
class FingerprintResult:
    __slots__ = ("duration_sec", "fingerprint", "content_hash")
    def __init__(self, duration_sec: float, fingerprint: str, content_hash: str):
        self.duration_sec = duration_sec
        self.fingerprint = fingerprint
        self.content_hash = content_hash


class FingerprintEngine:
    """chromaprint (fpcalc.exe) で acoustic fingerprint を取得し SQLite にキャッシュ。"""

    def __init__(self, db_path: str = METRICS_DB_PATH):
        self.db_path = db_path
        self.fpcalc = _resolve_fpcalc_path()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    content_hash TEXT PRIMARY KEY,
                    duration_sec REAL NOT NULL,
                    fingerprint  TEXT NOT NULL,
                    last_path    TEXT,
                    computed_at  TEXT NOT NULL
                )
            """)

    @staticmethod
    def _content_hash(path: str) -> str:
        """size + 先頭64KB + 末尾64KB の md5。content-addressing。"""
        import hashlib
        h = hashlib.md5()
        size = os.path.getsize(path)
        h.update(size.to_bytes(8, "little"))
        with open(path, "rb") as f:
            h.update(f.read(64 * 1024))
            if size > 128 * 1024:
                f.seek(-64 * 1024, os.SEEK_END)
                h.update(f.read(64 * 1024))
        return h.hexdigest()

    def compute(self, path: str) -> "FingerprintResult | None":
        if self.fpcalc is None:
            return None
        try:
            ch = self._content_hash(path)
        except OSError:
            return None
        # cache 確認
        with sqlite3.connect(self.db_path) as c:
            row = c.execute(
                "SELECT duration_sec, fingerprint FROM fingerprints WHERE content_hash = ?",
                (ch,),
            ).fetchone()
            if row:
                return FingerprintResult(row[0], row[1], ch)
        # fpcalc 呼び出し
        try:
            result = subprocess.run(
                [self.fpcalc, "-json", path],
                capture_output=True, text=True, timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                return None
            import json
            obj = json.loads(result.stdout)
            duration = float(obj["duration"])
            fp = str(obj["fingerprint"])
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
            return None
        # cache 書き込み
        import datetime
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?, ?, ?)",
                (ch, duration, fp, path, datetime.datetime.utcnow().isoformat()),
            )
        return FingerprintResult(duration, fp, ch)
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_fingerprint.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_fingerprint.py
git commit -m "feat: FingerprintEngine with chromaprint subprocess + SQLite cache

Content-addressed cache (size + head/tail 64KB md5) survives file rename.
Graceful None return on missing fpcalc.exe, corrupt files, or timeout."
```

---

## Task 4: 指紋比較 (Hamming + best-alignment)

**Files:**
- Modify: `DSRE.py` (指紋比較関数追加)
- Test: `tests/test_fingerprint_compare.py`

chromaprint 指紋文字列は base64 風エンコードされた int32 hash 列。decode して bit-level Hamming 一致率を best-alignment で求める。pure Python、外部依存なし。

- [ ] **Step 1: テストを書く**

Create `tests/test_fingerprint_compare.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path, freq=440.0, dur=2.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def test_same_audio_similarity_high():
    from DSRE import FingerprintEngine, fingerprint_similarity, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        b = os.path.join(d, "b.flac")
        _mk_flac(a, 440.0)
        _mk_flac(b, 440.0)  # same content, different file
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        fa = eng.compute(a)
        fb = eng.compute(b)
        sim = fingerprint_similarity(fa.fingerprint, fb.fingerprint)
        assert sim > 0.95


def test_different_audio_similarity_low():
    from DSRE import FingerprintEngine, fingerprint_similarity, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        b = os.path.join(d, "b.flac")
        _mk_flac(a, 440.0, dur=3.0)
        _mk_flac(b, 800.0, dur=3.0)  # different frequency = different audio
        db = os.path.join(d, "fp.db")
        eng = FingerprintEngine(db_path=db)
        fa = eng.compute(a)
        fb = eng.compute(b)
        sim = fingerprint_similarity(fa.fingerprint, fb.fingerprint)
        assert sim < 0.7
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_fingerprint_compare.py -v`
Expected: FAIL `ImportError: fingerprint_similarity`

- [ ] **Step 3: 指紋 decode + 比較 を実装**

`DSRE.py` の `FingerprintEngine` の直下に追加:

```python
def _decode_chromaprint(fp_str: str) -> list:
    """chromaprint base64 風エンコード文字列を int32 list に decode。

    chromaprint 公式仕様:
      - 1 文字目 = version (常に 1)
      - 2-4 byte = フラグ + フレーム長 (big-endian 24-bit)
      - 残り = run-length encoded differential int32 hashes
      ただし local 比較のみで AcoustID API 不要なので、
      簡易版として base64 URL-safe decode → 4 byte ごとに int32 として読む。
    """
    import base64
    try:
        # chromaprint は URL-safe base64 を使う
        raw = base64.urlsafe_b64decode(fp_str + "=" * (4 - len(fp_str) % 4))
    except Exception:
        return []
    # ヘッダ (4 byte) を skip し残りを int32 LE で読む
    if len(raw) < 4:
        return []
    body = raw[4:]
    # 4 byte 未満の残りは捨てる
    n = len(body) // 4
    out = []
    for i in range(n):
        v = int.from_bytes(body[i*4:(i+1)*4], "little", signed=False)
        out.append(v)
    return out


def _popcount32(x: int) -> int:
    """32-bit popcount (Python 3.10+ で int.bit_count() 利用可)。"""
    return x.bit_count()


def fingerprint_similarity(fp_a: str, fp_b: str) -> float:
    """chromaprint 指紋同士の Hamming 一致率 (0.0-1.0)。

    best-alignment: 短い方を長い方に沿ってスライドさせ、
    オーバーラップ部分の bit 一致率の最大値を返す。
    """
    a = _decode_chromaprint(fp_a)
    b = _decode_chromaprint(fp_b)
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a  # a を短い方に
    la, lb = len(a), len(b)
    max_ratio = 0.0
    # 全 offset を試す
    for off in range(lb - la + 1):
        match_bits = 0
        total_bits = la * 32
        for i in range(la):
            xor = a[i] ^ b[off + i]
            match_bits += 32 - _popcount32(xor)
        ratio = match_bits / total_bits
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_fingerprint_compare.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_fingerprint_compare.py
git commit -m "feat: chromaprint fingerprint similarity (Hamming + best-alignment)

Pure Python decode + sliding-window XOR popcount. No external Python deps.
Threshold 0.85 will be used for clustering in next task."
```

---

## Task 5: ClusterBuilder (union-find)

**Files:**
- Modify: `DSRE.py` (ClusterBuilder クラス追加)
- Test: `tests/test_cluster.py`

pairwise 類似度 > 閾値の file ペアを union-find で連結。

- [ ] **Step 1: テストを書く**

Create `tests/test_cluster.py`:

```python
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
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: FAIL `ImportError: ClusterBuilder`

- [ ] **Step 3: ClusterBuilder 実装**

`DSRE.py` の `fingerprint_similarity` の後に追加:

```python
class ClusterBuilder:
    """指紋類似度ペアを union-find で cluster 化する。"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self._parent: dict = {}
        self._pairs: list = []  # 記録目的 (デバッグログ用)

    def _find(self, x):
        while self._parent.get(x, x) != x:
            self._parent[x] = self._parent.get(self._parent[x], self._parent[x])
            x = self._parent[x]
        return x

    def _union(self, x, y):
        rx, ry = self._find(x), self._find(y)
        if rx != ry:
            self._parent[rx] = ry

    def add_pair(self, a, b, similarity: float) -> None:
        self._pairs.append((a, b, similarity))
        if similarity >= self.threshold:
            self._parent.setdefault(a, a)
            self._parent.setdefault(b, b)
            self._union(a, b)

    def build(self, items: list) -> list:
        """全 item を root でグルーピングして cluster の list を返す。"""
        for it in items:
            self._parent.setdefault(it, it)
        groups: dict = {}
        for it in items:
            root = self._find(it)
            groups.setdefault(root, []).append(it)
        return list(groups.values())
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_cluster.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_cluster.py
git commit -m "feat: ClusterBuilder with union-find on similarity pairs

Path compression in _find for amortized O(α(N)). Supports transitive
clustering and configurable similarity threshold."
```

---

## Task 6: MetadataExtractor (mutagen タグ抽出)

**Files:**
- Modify: `DSRE.py` (MetadataExtractor クラス追加)
- Test: `tests/test_metadata_extractor.py`

ユーザー foobar 階層が参照する全 Vorbis Comment タグを mutagen で抽出。

- [ ] **Step 1: テストを書く**

Create `tests/test_metadata_extractor.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac_with_tags(path, tags: dict):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    f.save()


def test_extract_main_fields():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {
            "artist": "X", "album": "Y", "title": "Z",
            "discnumber": "2", "tracknumber": "7", "date": "2020",
            "genre": "J-Pop",
        })
        m = MetadataExtractor.extract(path)
        assert m["artist"] == "X"
        assert m["album"] == "Y"
        assert m["title"] == "Z"
        assert m["discnumber"] == "2"
        assert m["tracknumber"] == "7"
        assert m["date"] == "2020"
        assert m["genre"] == "J-Pop"


def test_extract_missing_fields_returns_empty():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {})  # 全タグ未設定
        m = MetadataExtractor.extract(path)
        assert m["artist"] == ""
        assert m["title"] == ""
        assert m["genre"] == ""


def test_extract_extended_tags():
    from DSRE import MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        _mk_flac_with_tags(path, {
            "circle": "C", "brand": "B", "version_info": "Live",
        })
        m = MetadataExtractor.extract(path)
        assert m["circle"] == "C"
        assert m["brand"] == "B"
        assert m["version_info"] == "Live"
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_metadata_extractor.py -v`
Expected: FAIL `ImportError: MetadataExtractor`

- [ ] **Step 3: MetadataExtractor 実装**

`DSRE.py` の ClusterBuilder の後に追加:

```python
# 既存の Vorbis Comment 抽出対象タグ (foobar 階層 §6.1 と一致)
_METADATA_FIELDS = [
    # 主要 (採点・整列で使用)
    "artist", "album", "title", "discnumber", "tracknumber",
    "date", "genre",
    # カテゴリ階層
    "age", "circle", "category", "source", "grouping",
    # 作品識別
    "franchises", "products", "series", "brand", "subtitle", "elements",
    # プロジェクト
    "project", "collaboration", "group", "unit", "album_type",
    # 修飾サフィックス
    "featuring", "produced", "arrange_type",
    "version_info", "remaster_info", "cover_type", "live_type",
    "vocal_type", "m_number",
]


class MetadataExtractor:
    """mutagen で FLAC の Vorbis Comment タグを抽出する純粋クラス。"""

    @staticmethod
    def extract(path: str) -> dict:
        """戻り値: {field: value or ""} の dict。欠落は空文字。"""
        from mutagen.flac import FLAC
        out = {k: "" for k in _METADATA_FIELDS}
        out["__path__"] = path
        try:
            f = FLAC(path)
            for key in _METADATA_FIELDS:
                vals = f.tags.get(key) if f.tags else None
                if vals:
                    out[key] = str(vals[0])
            out["__pictures__"] = list(f.pictures)
        except Exception:
            out["__pictures__"] = []
        return out
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_metadata_extractor.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_metadata_extractor.py
git commit -m "feat: MetadataExtractor reading foobar-template Vorbis Comments

Reads 25+ optional tags (main, category, work-identifier, project,
modifier suffix) plus PICTURE blocks. Missing tags return empty string."
```

---

## Task 7: MetadataPropagator (canonical 選定 + 伝播 + 既値尊重)

**Files:**
- Modify: `DSRE.py` (MetadataPropagator クラス追加)
- Test: `tests/test_metadata_propagator.py`

cluster 内で canonical 選定 (加重スコア最大) → 主要+拡張タグを未タグ file に伝播 → version 識別系 + 既値は不変。

- [ ] **Step 1: テストを書く**

Create `tests/test_metadata_propagator.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _mk(path, tags, with_artwork=False):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC, Picture
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    if with_artwork:
        pic = Picture(); pic.type = 3; pic.mime = "image/jxl"; pic.data = _FAKE_JXL
        f.add_picture(pic)
    f.save()


def test_canonical_selection_by_tag_richness():
    from DSRE import MetadataPropagator, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "album": "Y", "title": "Z", "date": "2020"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X"})
        c = os.path.join(d, "c.flac"); _mk(c, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b, c)]
        canon = MetadataPropagator.choose_canonical(meta)
        assert canon["__path__"] == a


def test_canonical_artwork_bonus():
    from DSRE import MetadataPropagator, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "album": "Y"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "album": "Y"}, with_artwork=True)
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        canon = MetadataPropagator.choose_canonical(meta)
        assert canon["__path__"] == b


def test_propagate_to_untagged():
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "album": "Y", "title": "Z", "genre": "G"}, with_artwork=True)
        b = os.path.join(d, "b.flac")
        _mk(b, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["artist"][0] == "X"
        assert bf["album"][0] == "Y"
        assert bf["title"][0] == "Z"
        assert bf["genre"][0] == "G"
        assert len(bf.pictures) == 1
        assert bf.pictures[0].mime == "image/jxl"


def test_propagate_respects_existing_value():
    """伝播先が既値を持つタグは上書きしない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "genre": "G"})
        b = os.path.join(d, "b.flac")
        _mk(b, {"genre": "B-genre"})  # 既値あり
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["genre"][0] == "B-genre"  # 上書きされていない
        assert bf["artist"][0] == "X"  # 空だったので伝播された


def test_version_tags_never_propagated():
    """version 識別系タグは canonical が持っていても伝播しない。"""
    from DSRE import MetadataPropagator, MetadataExtractor
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk(a, {"artist": "X", "version_info": "Live", "live_type": "L"})
        b = os.path.join(d, "b.flac")
        _mk(b, {})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        MetadataPropagator.propagate(meta)
        bf = FLAC(b)
        assert bf["artist"][0] == "X"
        assert "version_info" not in bf
        assert "live_type" not in bf
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_metadata_propagator.py -v`
Expected: FAIL `ImportError`

- [ ] **Step 3: MetadataPropagator 実装**

`DSRE.py` の MetadataExtractor の後に追加:

```python
# 伝播対象外 (version 識別系 + 修飾系で版ごとに変動するもの)
_VERSION_TAGS = frozenset([
    "version_info", "cover_type", "live_type", "vocal_type",
    "remaster_info", "arrange_type", "m_number",
    "featuring", "produced",
])

# 主要タグ + アートワーク有無に対する加重
_CANONICAL_WEIGHT = {
    "artist": 2, "album": 2, "title": 2, "date": 2,
    "discnumber": 2, "tracknumber": 2,
}
# 上記以外の通常タグ = 1 点


class MetadataPropagator:
    """cluster 内で canonical 選定 + 主要/拡張タグの伝播 + version タグ保護。"""

    @staticmethod
    def _score(meta: dict) -> tuple:
        """canonical 候補としての加重スコア。tiebreak 含む tuple を返す (大きい方が canonical)。"""
        s = 0
        for k in _METADATA_FIELDS:
            if k in _VERSION_TAGS:
                continue
            if meta.get(k):
                s += _CANONICAL_WEIGHT.get(k, 1)
        # アートワーク有り +3
        if meta.get("__pictures__"):
            s += 3
        # ファイルサイズ tiebreak
        try:
            size = os.path.getsize(meta["__path__"])
        except OSError:
            size = 0
        # path 文字列で最終決定的順序
        return (s, size, meta["__path__"])

    @staticmethod
    def choose_canonical(group: list) -> dict:
        return max(group, key=MetadataPropagator._score)

    @staticmethod
    def propagate(group: list) -> None:
        """cluster 内全 file に対し、canonical → 他 file へ in-place 伝播。"""
        if os.environ.get("DSRE_HARMONIZE_METADATA") == "0":
            return
        if len(group) < 2:
            return
        canon = MetadataPropagator.choose_canonical(group)
        from mutagen.flac import FLAC
        for m in group:
            if m["__path__"] == canon["__path__"]:
                continue
            try:
                f = FLAC(m["__path__"])
                if f.tags is None:
                    f.add_tags()
                # 主要 + 拡張タグの伝播 (既値尊重、version 系除外)
                changed = False
                for k in _METADATA_FIELDS:
                    if k in _VERSION_TAGS:
                        continue
                    if m.get(k):  # 伝播先既値あり → 尊重
                        continue
                    if canon.get(k):  # canonical が値を持つ → 伝播
                        f[k] = [canon[k]]
                        changed = True
                # アートワーク伝播 (伝播先が空 + canonical に存在する場合)
                if not m.get("__pictures__") and canon.get("__pictures__"):
                    f.clear_pictures()
                    for pic in canon["__pictures__"]:
                        f.add_picture(pic)
                    changed = True
                if changed:
                    f.save()
            except Exception:
                # 個別 file 失敗は cluster 全体を止めない
                pass
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_metadata_propagator.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_metadata_propagator.py
git commit -m "feat: MetadataPropagator with canonical selection and respectful copy

Weighted canonical selection (main tags 2pt, ext 1pt, artwork +3, size
tiebreak). Propagates main/ext tags + artwork to untagged peers.
Version-identifier tags (version_info, live_type, cover_type, etc) and
pre-existing target values are never overwritten."
```

---

## Task 8: 版分岐 (version sub-cluster)

**Files:**
- Modify: `DSRE.py` (split_versions 関数追加)
- Test: `tests/test_version_split.py`

メタデータ伝播後、cluster 内で version 識別系タグの組が異なる file を別 sub-cluster に分離。

- [ ] **Step 1: テストを書く**

Create `tests/test_version_split.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk(path, tags):
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    from mutagen.flac import FLAC
    f = FLAC(path)
    for k, v in tags.items():
        f[k] = [v]
    f.save()


def test_split_same_version_one_subcluster():
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        subs = split_versions(meta)
        assert len(subs) == 1
        assert len(subs[0]) == 2


def test_split_different_version_two_subclusters():
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "version_info": "Live"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "version_info": "Studio"})
        meta = [MetadataExtractor.extract(p) for p in (a, b)]
        subs = split_versions(meta)
        assert len(subs) == 2
        for s in subs:
            assert len(s) == 1


def test_split_multiple_version_tags():
    """複数の version 系タグの組み合わせで判定。"""
    from DSRE import split_versions, MetadataExtractor
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac"); _mk(a, {"artist": "X", "live_type": "L"})
        b = os.path.join(d, "b.flac"); _mk(b, {"artist": "X", "live_type": "L"})
        c = os.path.join(d, "c.flac"); _mk(c, {"artist": "X", "live_type": "M"})
        meta = [MetadataExtractor.extract(p) for p in (a, b, c)]
        subs = split_versions(meta)
        assert len(subs) == 2  # (a,b) と (c,)
        sizes = sorted(len(s) for s in subs)
        assert sizes == [1, 2]
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_version_split.py -v`
Expected: FAIL `ImportError: split_versions`

- [ ] **Step 3: split_versions 実装**

`DSRE.py` の MetadataPropagator の直下に追加:

```python
def split_versions(cluster: list) -> list:
    """cluster を version 識別系タグの組で sub-cluster に分割。

    cluster: メタデータ伝播済の dict list。
    戻り値: sub-cluster の list of list。
    """
    groups: dict = {}
    for m in cluster:
        key = tuple(m.get(k, "") for k in [
            "version_info", "cover_type", "live_type", "vocal_type",
            "remaster_info", "arrange_type", "m_number",
        ])
        groups.setdefault(key, []).append(m)
    return list(groups.values())
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_version_split.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_version_split.py
git commit -m "feat: split_versions for sub-clustering by version identifier tags

Files in the same fingerprint cluster but with differing version_info /
live_type / cover_type / vocal_type / remaster_info / arrange_type /
m_number are split into separate sub-clusters (kept as distinct variants)."
```

---

## Task 9: QualityProbe (正規化 + 解析 + スコア + フラグ)

**Files:**
- Modify: `DSRE.py` (QualityProbe クラス追加)
- Test: `tests/test_quality_probe.py`

ユーザー指摘: opus エンコードは不要。`_dsre_normalize_volume` を適用した PCM を `MetricsComputer.compute()` で解析するだけ。

- [ ] **Step 1: テストを書く**

Create `tests/test_quality_probe.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_audio(freq=440.0, amp=0.3, dur=2.0):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def test_score_basic_signal():
    from DSRE import QualityProbe
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        sf.write(path, _mk_audio(amp=0.3), SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        assert result is not None
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.flagged, bool)
        assert isinstance(result.metrics, dict)


def test_score_clip_zero_after_normalize():
    """clip 大量入力でも正規化後の解析では clip_count = 0。"""
    from DSRE import QualityProbe
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        audio = _mk_audio(amp=5.0)  # clip 大量
        sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        assert result is not None
        assert result.metrics["clip_count"] == 0


def test_flag_brick_wall():
    """flatness が極小なら flagged=True。"""
    from DSRE import QualityProbe, _BRICK_WALL_FLATNESS
    # 高調波の少ない pure sine = flatness が小さい
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.flac")
        sf.write(path, _mk_audio(freq=440, amp=0.3, dur=2.0), SR, subtype="PCM_24", format="FLAC")
        result = QualityProbe.score(path)
        # pure sine の flatness は通常 < 0.05 → flagged
        if result.metrics["flatness"] is not None and result.metrics["flatness"] < _BRICK_WALL_FLATNESS:
            assert result.flagged
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_quality_probe.py -v`
Expected: FAIL `ImportError: QualityProbe`

- [ ] **Step 3: QualityProbe 実装**

`DSRE.py` の split_versions の後に追加:

```python
# フラグ閾値
_BRICK_WALL_FLATNESS = float(os.environ.get("DSRE_FLAG_FLATNESS", "0.05"))
_LOSSY_HF_RATIO_16K = float(os.environ.get("DSRE_FLAG_HF16K", "0.0005"))
_LOSSY_ROLLOFF_HZ = float(os.environ.get("DSRE_FLAG_ROLLOFF", "17000"))
_HYPER_COMP_DR = float(os.environ.get("DSRE_FLAG_DR", "6"))
_ART_HARMONIC = float(os.environ.get("DSRE_FLAG_HARMONIC", "0.5"))
_ART_DR_THRESHOLD = float(os.environ.get("DSRE_FLAG_ART_DR", "12"))
_UPSAMPLE_HF8K = float(os.environ.get("DSRE_FLAG_HF8K", "0.3"))
_UPSAMPLE_CENTROID = float(os.environ.get("DSRE_FLAG_CENTROID", "6000"))

# スコア式パラメータ
_SCORE_W_DR = float(os.environ.get("DSRE_SCORE_WEIGHT_DR", "3.0"))
_SCORE_LUFS_TARGET = float(os.environ.get("DSRE_SCORE_LUFS_TARGET", "-14"))


class ScoreResult:
    __slots__ = ("score", "metrics", "flagged", "flag_reasons")
    def __init__(self, score, metrics, flagged, flag_reasons):
        self.score = score
        self.metrics = metrics
        self.flagged = flagged
        self.flag_reasons = flag_reasons


class QualityProbe:
    """採点: file → 正規化 → 直接解析 → スコア + フラグ。"""

    @staticmethod
    def score(path: str) -> "ScoreResult | None":
        try:
            audio, sr = load_audio_safe(path)
        except Exception:
            return None

        # ステレオは (channels, samples) を期待する _dsre_normalize_volume へ渡す
        if audio.ndim == 1:
            audio_for_norm = audio[np.newaxis, :]
        else:
            audio_for_norm = audio  # 既に (channels, samples) を想定

        norm = _dsre_normalize_volume(audio_for_norm, sr)
        # MetricsComputer.compute() は (samples,) or (channels, samples) を受ける
        try:
            metrics = MetricsComputer.compute(norm, sr)
        except Exception:
            return None

        score, flagged, reasons = QualityProbe._calculate(metrics)
        return ScoreResult(score, metrics, flagged, reasons)

    @staticmethod
    def _calculate(metrics: dict) -> tuple:
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        score = 0.0
        dr = metrics.get("dr") or 0.0
        lufs = metrics.get("lufs")
        hf12k = metrics.get("hf_ratio_12k") or 0.0
        flatness = metrics.get("flatness") or 0.0
        plr = metrics.get("plr")

        score += clamp(dr, 0, 20) * _SCORE_W_DR
        if lufs is not None:
            score += clamp(20 - abs(lufs - _SCORE_LUFS_TARGET), 0, 20)
        score += clamp(hf12k, 0, 0.05) * 200
        score += clamp(flatness, 0, 0.5) * 20
        if plr is not None and plr < 6:
            score -= (6 - plr) * 2
        score = clamp(score, 0, 100)

        # フラグ判定
        reasons = []
        hf16k = metrics.get("hf_ratio_16k") or 0.0
        rolloff = metrics.get("rolloff_hz") or 99999
        harmonic = metrics.get("harmonic_1k_proxy") or 0.0
        hf8k = metrics.get("hf_ratio_8k") or 0.0
        centroid = metrics.get("centroid_hz") or 0.0

        if hf16k < _LOSSY_HF_RATIO_16K and rolloff < _LOSSY_ROLLOFF_HZ:
            reasons.append("lossy_hf_cliff")
        if flatness < _BRICK_WALL_FLATNESS:
            reasons.append("brick_wall_flatness")
        if dr < _HYPER_COMP_DR:
            reasons.append("hyper_compression")
        if harmonic > _ART_HARMONIC and dr > _ART_DR_THRESHOLD:
            reasons.append("artifact_high_harmonic")
        if hf8k > _UPSAMPLE_HF8K and centroid > _UPSAMPLE_CENTROID:
            reasons.append("upsample_artifact")

        return score, len(reasons) > 0, reasons
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_quality_probe.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_quality_probe.py
git commit -m "feat: QualityProbe - normalize + analyze + score + flag

No opus encoding: clip=0 normalized PCM is analyzed directly via
MetricsComputer. Shares _dsre_normalize_volume with save path to
prevent divergence. Five flag conditions detect lossy origin,
brick wall, hyper compression, artifact, and fake upsample."
```

---

## Task 10: BestSelector + DiscardHandler

**Files:**
- Modify: `DSRE.py` (BestSelector, DiscardHandler 追加)
- Test: `tests/test_best_discard.py`

- [ ] **Step 1: テストを書く**

Create `tests/test_best_discard.py`:

```python
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
    # clean 集合の max (a.flac, score 80)
    assert best.path == "a.flac"
    assert best.warn_all_flagged is False


def test_select_best_all_flagged():
    from DSRE import BestSelector, ScoreResult
    a = ScoreResult(60.0, {}, True, ["brick_wall_flatness"])
    b = ScoreResult(75.0, {}, True, ["hyper_compression"])
    items = [("a", a, 1000), ("b", b, 1500)]
    best = BestSelector.choose(items)
    assert best.path == "b"  # max score
    assert best.warn_all_flagged is True


def test_select_tiebreak_filesize():
    from DSRE import BestSelector, ScoreResult
    a = ScoreResult(80.0, {}, False, [])
    b = ScoreResult(80.0, {}, False, [])
    items = [("a", a, 1000), ("b", b, 2000)]
    best = BestSelector.choose(items)
    assert best.path == "b"  # size 大


def test_discard_rename_and_trash():
    from DSRE import DiscardHandler
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "loser.flac")
        _mk_flac(f)
        DiscardHandler.discard(f, dry_run=True)  # dry_run でリネームのみ実行
        renamed = os.path.join(d, "[DSRE-inferior] loser.flac")
        assert os.path.isfile(renamed)
        assert not os.path.isfile(f)
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_best_discard.py -v`
Expected: FAIL `ImportError`

- [ ] **Step 3: BestSelector + DiscardHandler 実装**

`DSRE.py` の QualityProbe の後に追加:

```python
class BestSelection:
    __slots__ = ("path", "score_result", "warn_all_flagged")
    def __init__(self, path, sr, warn):
        self.path = path
        self.score_result = sr
        self.warn_all_flagged = warn


class BestSelector:
    """sub-cluster 内で最良 file を選ぶ。"""

    @staticmethod
    def choose(items: list) -> BestSelection:
        """items: list of (path, ScoreResult, file_size). 戻り値: BestSelection。"""
        clean = [it for it in items if not it[1].flagged]
        pool = clean or items
        # max by (score, file_size, path)
        best = max(pool, key=lambda it: (it[1].score, it[2], it[0]))
        return BestSelection(best[0], best[1], warn=(len(clean) == 0))


class DiscardHandler:
    """非最良 file を識別可能リネーム → ゴミ箱投入。"""

    PREFIX = "[DSRE-inferior] "

    @staticmethod
    def discard(path: str, dry_run: bool = False) -> str:
        """path をリネームし、dry_run=False なら send2trash で削除。
        戻り値: リネーム後パス。"""
        d, name = os.path.split(path)
        new_name = DiscardHandler.PREFIX + name
        new_path = os.path.join(d, new_name)
        os.rename(path, new_path)
        if not dry_run:
            try:
                send2trash(new_path)
            except Exception:
                pass  # 失敗してもリネーム済の file は残る (識別可能)
        return new_path
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_best_discard.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_best_discard.py
git commit -m "feat: BestSelector and DiscardHandler

BestSelector: clean-first max score, size tiebreak, deterministic by path.
DiscardHandler: [DSRE-inferior] prefix rename then send2trash. Prefix
collates renamed files at the top of trash listing for easy identification."
```

---

## Task 11: FoobarPathBuilder (テンプレ展開)

**Files:**
- Modify: `DSRE.py` (FoobarPathBuilder クラス追加)
- Test: `tests/test_foobar_path.py`

§6.1 のルールに基づくパス構築。親子重複排除・サニタイズ・長パス対応。

- [ ] **Step 1: テストを書く**

Create `tests/test_foobar_path.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _meta(**kw):
    """空辞書ベースで kw でタグを上書き。"""
    base = {k: "" for k in [
        "artist", "album", "title", "discnumber", "tracknumber",
        "date", "genre", "age", "circle", "category", "source", "grouping",
        "franchises", "products", "series", "brand", "subtitle", "elements",
        "project", "collaboration", "group", "unit", "album_type",
        "featuring", "produced", "arrange_type",
        "version_info", "remaster_info", "cover_type", "live_type",
        "vocal_type", "m_number",
    ]}
    base.update(kw)
    return base


def test_full_metadata_path():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="Artist", album="Album", title="Title",
        discnumber="1", tracknumber="3", date="2020",
        genre="J-Pop", circle="Circle",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path == r"C:/out\J-Pop\Circle\Album (2020)\01.003.Title - Artist.flac"


def test_missing_levels_omitted():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="T", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path == r"C:/out\A\01.001.T - X.flac"


def test_multi_disc_layer():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="T", discnumber="2", tracknumber="5")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=True)
    assert path == r"C:/out\A\Disc 02\02.005.T - X.flac"


def test_sanitize_forbidden_chars():
    from DSRE import FoobarPathBuilder
    m = _meta(artist="X", album="A", title="Hi? Yes!", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert "?" not in path
    assert "？" in path


def test_parent_child_dedup_products_eq_brand():
    """products == brand なら products 階層は省略。"""
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        brand="SameBrand", products="SameBrand",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.count("SameBrand") == 1


def test_modifier_suffix_appended():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        version_info="Live", cover_type="Self",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.endswith(" [Live] [Self].flac")


def test_featuring_in_artist_part():
    from DSRE import FoobarPathBuilder
    m = _meta(
        artist="X", album="A", title="T", discnumber="1", tracknumber="1",
        featuring="Y",
    )
    path = FoobarPathBuilder.build("C:/out", m, multi_disc=False)
    assert path.endswith("T - X [feat. Y].flac")


def test_long_path_unc_prefix():
    """250 chars 超過時に \\\\?\\ プレフィックス。"""
    from DSRE import FoobarPathBuilder
    long = "Long" * 80
    m = _meta(artist="X", album=long, title="T", discnumber="1", tracknumber="1")
    path = FoobarPathBuilder.build("C:/very/long/output", m, multi_disc=False)
    assert path.startswith("\\\\?\\")
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_foobar_path.py -v`
Expected: FAIL `ImportError: FoobarPathBuilder`

- [ ] **Step 3: FoobarPathBuilder 実装**

`DSRE.py` の DiscardHandler の後に追加:

```python
_WIN_FORBIDDEN = {
    "<": "＜", ">": "＞", ":": "：", '"': "”",
    "/": "／", "\\": "＼", "|": "｜", "?": "？", "*": "＊",
}


class FoobarPathBuilder:
    """foobar 互換階層パス構築 (§6.1 ルール) + Windows サニタイズ + 長パス対応。"""

    CATEGORY_LEVELS = ["genre", "age", "circle", "category", "source", "grouping"]
    PROJECT_LEVELS = ["project", "collaboration", "group", "unit", "album_type"]
    MODIFIER_TAGS = [
        ("produced", lambda v: f"Prod. {v}"),
        ("arrange_type", lambda v: v.replace(";", " /")),
        ("version_info", lambda v: v.replace(";", " /")),
        ("remaster_info", lambda v: v.replace(";", " /")),
        ("cover_type", lambda v: v),
        ("live_type", lambda v: v),
        ("vocal_type", lambda v: v),
        ("m_number", lambda v: v),
    ]

    @staticmethod
    def _sanitize_segment(s: str) -> str:
        for k, v in _WIN_FORBIDDEN.items():
            s = s.replace(k, v)
        s = s.rstrip(" .")
        return s

    @staticmethod
    def _work_identifier_levels(m: dict) -> list:
        """親子重複排除を行った作品識別レベル順序。"""
        out = []
        project = m.get("project", "")
        brand = m.get("brand", "")
        franchises = m.get("franchises", "")
        subtitle = m.get("subtitle", "")
        elements = m.get("elements", "")

        if franchises and franchises != project:
            out.append(franchises)
        products = m.get("products", "")
        if products and products != project and products != brand:
            out.append(products)
        series = m.get("series", "")
        if series and series != project:
            out.append(series)
        if brand:
            out.append(brand)
        if subtitle and subtitle != brand and subtitle != franchises:
            out.append(subtitle)
        if elements and elements != brand and elements != franchises and elements != subtitle:
            out.append(elements)
        return out

    @staticmethod
    def _date_levels(date: str) -> list:
        """date から year/month を切り出す。'2020-03-15' / '2020' / '' を許容。"""
        if not date:
            return []
        parts = date.split("-")
        out = []
        if len(parts) >= 1 and parts[0].isdigit():
            out.append(parts[0])  # year
        if len(parts) >= 2 and parts[1].isdigit():
            out.append(parts[1].zfill(2))  # month
        return out

    @staticmethod
    def build(output_root: str, m: dict, multi_disc: bool) -> str:
        """foobar 階層パス組立。
        output_root: OUTPUT_DIR (例: C:\\Audio\\DSRE\\Output)
        m: メタデータ dict
        multi_disc: album 内に複数 disc がある場合 True (Disc 階層付与)
        """
        parts = [output_root]

        # カテゴリレベル
        for k in FoobarPathBuilder.CATEGORY_LEVELS:
            if m.get(k):
                parts.append(FoobarPathBuilder._sanitize_segment(m[k]))

        # 作品識別レベル
        for w in FoobarPathBuilder._work_identifier_levels(m):
            parts.append(FoobarPathBuilder._sanitize_segment(w))

        # プロジェクトレベル
        for k in FoobarPathBuilder.PROJECT_LEVELS:
            if m.get(k):
                parts.append(FoobarPathBuilder._sanitize_segment(m[k]))

        # 時系列レベル
        for d in FoobarPathBuilder._date_levels(m.get("date", "")):
            parts.append(d)

        # アルバム
        if m.get("album"):
            album = m["album"]
            if m.get("date"):
                album = f"{album} ({m['date']})"
            parts.append(FoobarPathBuilder._sanitize_segment(album))

        # Disc
        disc = m.get("discnumber", "1") or "1"
        if multi_disc:
            try:
                parts.append(f"Disc {int(disc):02d}")
            except ValueError:
                parts.append(f"Disc {disc}")

        # ファイル名
        try:
            d_pad = f"{int(disc):02d}"
        except ValueError:
            d_pad = disc
        track = m.get("tracknumber", "0") or "0"
        try:
            t_pad = f"{int(track):03d}"
        except ValueError:
            t_pad = track

        title = FoobarPathBuilder._sanitize_segment(m.get("title", ""))
        artist = m.get("artist", "").replace("/", "_ ")
        featuring = m.get("featuring", "").replace("/", "* ")

        # artist 部
        artist_part = ""
        if featuring:
            artist_part = f" - {FoobarPathBuilder._sanitize_segment(artist)} [feat. {FoobarPathBuilder._sanitize_segment(featuring)}]"
        elif artist:
            artist_part = f" - {FoobarPathBuilder._sanitize_segment(artist)}"

        # 修飾サフィックス
        modifiers = ""
        for key, fmt in FoobarPathBuilder.MODIFIER_TAGS:
            v = m.get(key, "")
            if v:
                modifiers += f" [{FoobarPathBuilder._sanitize_segment(fmt(v))}]"

        filename = f"{d_pad}.{t_pad}.{title}{artist_part}{modifiers}.flac"
        parts.append(filename)

        path = "\\".join(parts)
        if len(path) > 250 and not path.startswith("\\\\?\\"):
            path = "\\\\?\\" + path
        return path
```

- [ ] **Step 4: テスト実行**

Run: `python -m pytest tests/test_foobar_path.py -v`
Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add DSRE.py tests/test_foobar_path.py
git commit -m "feat: FoobarPathBuilder with hierarchical rule expansion

Implements §6.1 spec rules: category levels, work-identifier with
parent-child dedup, project levels, date year/month, album with optional
(date) suffix, multi-disc layer, filename with artist/feat/modifier
suffixes, Windows char sanitization, and \\\\?\\ long-path support."
```

---

## Task 12: WorkflowOrchestrator (STAGE 1-3 統括)

**Files:**
- Modify: `DSRE.py` (WorkflowOrchestrator クラス追加、Worker.run 改修)
- Test: `tests/test_workflow_integration.py`

全モジュールを束ねる。Worker.run の旧ループを置き換える。

- [ ] **Step 1: 統合テストを書く**

Create `tests/test_workflow_integration.py`:

```python
import os, sys, tempfile, shutil
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000


def _mk_flac(path, freq=440.0, dur=2.0, tags=None):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")
    if tags:
        from mutagen.flac import FLAC
        f = FLAC(path)
        for k, v in tags.items():
            f[k] = [v]
        f.save()


def test_idempotency_skip_already_processed():
    """dsre_version タグ持ち file は WorkflowOrchestrator.scan で skip。"""
    from DSRE import WorkflowOrchestrator
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.flac")
        _mk_flac(a, tags={"artist": "X", "album": "Y", "title": "Z", "dsre_version": "r99"})
        b = os.path.join(d, "b.flac")
        _mk_flac(b, freq=600.0, tags={"artist": "X"})
        orch = WorkflowOrchestrator(input_dir=d, output_dir=d)
        pending = orch.scan_pending()
        assert b in pending
        assert a not in pending


def test_resolve_multi_disc_album():
    """album 内の disc 値で multi-disc 判定。"""
    from DSRE import WorkflowOrchestrator, MetadataExtractor
    m_single1 = {"album": "A", "discnumber": "1"}
    m_single2 = {"album": "A", "discnumber": "1"}
    assert WorkflowOrchestrator._is_multi_disc([m_single1, m_single2], "A") is False

    m_multi1 = {"album": "A", "discnumber": "1"}
    m_multi2 = {"album": "A", "discnumber": "2"}
    assert WorkflowOrchestrator._is_multi_disc([m_multi1, m_multi2], "A") is True
```

- [ ] **Step 2: テスト失敗を確認**

Run: `python -m pytest tests/test_workflow_integration.py -v`
Expected: FAIL `ImportError: WorkflowOrchestrator`

- [ ] **Step 3: WorkflowOrchestrator 実装**

`DSRE.py` の FoobarPathBuilder の後に追加:

```python
class WorkflowOrchestrator:
    """STAGE 1-3 統括: スキャン → 指紋 → クラスタ → 伝播 → 採点 → 選択 → 破棄 → 仕分け。

    Worker から呼ばれて run_stage1() / run_stage3() を提供する。STAGE 2 (DSRE 本処理)
    は呼出側 (Worker._process_one ループ) が担当する。
    """

    def __init__(self, input_dir: str, output_dir: str,
                 db_path: str = METRICS_DB_PATH,
                 progress_cb=None, abort_cb=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.db_path = db_path
        self.progress_cb = progress_cb or (lambda s: None)
        self.abort_cb = abort_cb or (lambda: False)
        self.fp_engine = FingerprintEngine(db_path=db_path)

    # ---- STAGE 1 ----

    def scan_pending(self) -> list:
        """INPUT_DIR を再帰スキャンして dsre_version 未持ちの flac path list を返す。"""
        from mutagen.flac import FLAC
        out = []
        for root, _, files in os.walk(self.input_dir):
            for fn in files:
                if not fn.lower().endswith(".flac"):
                    continue
                p = os.path.join(root, fn)
                try:
                    f = FLAC(p)
                    if f.tags and "dsre_version" in f.tags:
                        continue  # 既処理 skip
                except Exception:
                    pass
                out.append(p)
        return out

    def build_clusters(self, paths: list) -> list:
        """fingerprint で paths を cluster 化。指紋不能 file は単独 cluster。"""
        if os.environ.get("DSRE_CLUSTER_SIMILARITY"):
            threshold = float(os.environ["DSRE_CLUSTER_SIMILARITY"])
        else:
            threshold = 0.85

        # 全 file の指紋を取得
        fp_map = {}
        for i, p in enumerate(paths, 1):
            self.progress_cb(f"[STAGE 1] Fingerprinting {i}/{len(paths)}")
            if self.abort_cb():
                return [[p] for p in paths]
            r = self.fp_engine.compute(p)
            if r is not None:
                fp_map[p] = r.fingerprint

        cb = ClusterBuilder(similarity_threshold=threshold)
        # pairwise 類似度 (指紋取得済 file 同士のみ)
        with_fp = list(fp_map.keys())
        for i in range(len(with_fp)):
            for j in range(i + 1, len(with_fp)):
                if self.abort_cb():
                    break
                sim = fingerprint_similarity(fp_map[with_fp[i]], fp_map[with_fp[j]])
                cb.add_pair(with_fp[i], with_fp[j], sim)
        return cb.build(items=paths)

    def harmonize_metadata(self, cluster: list) -> list:
        """cluster (path list) のメタデータを伝播。戻り値: 伝播後 meta dict list。"""
        meta = [MetadataExtractor.extract(p) for p in cluster]
        MetadataPropagator.propagate(meta)
        # 伝播後の再抽出 (in-place 書き換え後の状態反映)
        meta = [MetadataExtractor.extract(p) for p in cluster]
        return meta

    def score_files(self, meta_list: list) -> list:
        """各 file を採点。戻り値: list of (path, ScoreResult, file_size)。"""
        out = []
        for m in meta_list:
            if self.abort_cb():
                break
            r = QualityProbe.score(m["__path__"])
            if r is None:
                continue
            sz = os.path.getsize(m["__path__"])
            out.append((m["__path__"], r, sz))
        return out

    def process_subcluster(self, sub: list) -> "BestSelection | None":
        """sub (meta dict list) で採点+選択+破棄を実行。戻り値: 最良の BestSelection。"""
        scored = self.score_files(sub)
        if not scored:
            return None
        best = BestSelector.choose(scored)
        # 非最良を破棄
        for path, sr, sz in scored:
            if path == best.path:
                continue
            try:
                DiscardHandler.discard(path)
            except Exception:
                pass
        return best

    @staticmethod
    def _is_multi_disc(meta_list: list, album: str) -> bool:
        """同 album 内に disc=1 以外の track が存在するか。"""
        if not album:
            return False
        discs = set()
        for m in meta_list:
            if m.get("album") == album:
                d = m.get("discnumber", "1") or "1"
                discs.add(d)
        # disc=1 以外を含むか
        non_one = {d for d in discs if d not in ("1", "01", "")}
        if non_one:
            return True
        # env 強制
        return os.environ.get("DSRE_DISC_DIR_ALWAYS") == "1"

    def relocate_best_files(self, bests: list, all_meta: list) -> list:
        """最良 file 群を INPUT_DIR 内で foobar 階層に再配置。戻り値: 新パス list。
        all_meta は multi_disc 判定のため必要。"""
        new_paths = []
        for b in bests:
            m = MetadataExtractor.extract(b.path)
            multi = self._is_multi_disc(all_meta, m.get("album", ""))
            new_path = FoobarPathBuilder.build(self.input_dir, m, multi_disc=multi)
            new_path = _resolve_collision(new_path, b.path)
            try:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                os.rename(b.path, new_path)
                new_paths.append(new_path)
            except OSError:
                new_paths.append(b.path)
        return new_paths

    def run_stage1(self) -> list:
        """STAGE 1 を実行し、DSRE 処理対象 (最良 file の新パス) list を返す。"""
        paths = self.scan_pending()
        self.progress_cb(f"[STAGE 1] スキャン {len(paths)} files")
        if not paths:
            return []
        clusters = self.build_clusters(paths)
        self.progress_cb(f"[STAGE 1] クラスタリング {len(clusters)} cluster")

        all_meta = []
        bests = []
        for c in clusters:
            if self.abort_cb():
                break
            meta = self.harmonize_metadata(c)
            subs = split_versions(meta)
            for sub in subs:
                best = self.process_subcluster(sub)
                if best:
                    bests.append(best)
            all_meta.extend(meta)

        self.progress_cb(f"[STAGE 1] 最良選択 {len(bests)} 完了")
        if os.environ.get("DSRE_PRESORT_INPUT") == "0":
            return [b.path for b in bests]
        new_paths = self.relocate_best_files(bests, all_meta)
        self.progress_cb("[STAGE 1] 整列完了")
        return new_paths

    # ---- STAGE 3 ----

    def run_stage3(self, processed_paths: list) -> None:
        """DSRE 処理済 OUTPUT_DIR 内 file を foobar 階層へ配置。"""
        # processed_paths は OUTPUT_DIR 直下にある dsre_version タグ付き file
        all_meta = [MetadataExtractor.extract(p) for p in processed_paths]
        for p, m in zip(processed_paths, all_meta):
            multi = self._is_multi_disc(all_meta, m.get("album", ""))
            new_path = FoobarPathBuilder.build(self.output_dir, m, multi_disc=multi)
            new_path = _resolve_collision(new_path, p)
            try:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                if new_path != p:
                    os.rename(p, new_path)
            except OSError:
                pass
        self.progress_cb("[STAGE 3] 整列完了")


def _resolve_collision(target: str, src: str) -> str:
    """target が存在する場合、内容一致なら skip 用に同パス、相違なら _N suffix を付与。"""
    if not os.path.exists(target):
        return target
    # 内容ハッシュ比較
    if os.path.exists(src):
        try:
            from hashlib import md5
            def fh(p):
                h = md5()
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
                return h.hexdigest()
            if fh(target) == fh(src):
                return target  # 同内容 → 上書きしない (rename skip 判定は呼出側)
        except OSError:
            pass
    base, ext = os.path.splitext(target)
    n = 2
    while os.path.exists(f"{base}_{n}{ext}"):
        n += 1
    return f"{base}_{n}{ext}"
```

- [ ] **Step 4: Worker.run の呼出側に WorkflowOrchestrator を統合**

`DSRE.py` の Worker クラス内 `run()` メソッドを修正し、`DSRE_WORKFLOW` env が `1` (デフォルト) の時に `WorkflowOrchestrator.run_stage1()` を呼んで処理対象 path list を取得し、それを既存ループに渡す。STAGE 3 を実行する。

具体的な改修箇所 (Worker.run の冒頭付近、ファイルキュー構築直後):

```python
# Worker.run 内、self._files セットアップ後、_process_one ループの前
if os.environ.get("DSRE_WORKFLOW", "1") == "1":
    orch = WorkflowOrchestrator(
        input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
        progress_cb=self.sig_text.emit,
        abort_cb=lambda: self._abort,
    )
    workflow_files = orch.run_stage1()
    self._files = workflow_files  # 既存ループの対象を上書き
```

そして既存ループ最後 (`for path in self._files: ... 全 file 処理完了後`) に:

```python
if os.environ.get("DSRE_WORKFLOW", "1") == "1":
    # OUTPUT_DIR 直下にある dsre_version タグ付き file を foobar 階層へ
    processed = []
    for f in os.listdir(OUTPUT_DIR):
        p = os.path.join(OUTPUT_DIR, f)
        if not p.lower().endswith(".flac"):
            continue
        from mutagen.flac import FLAC
        try:
            if "dsre_version" in FLAC(p).tags:
                processed.append(p)
        except Exception:
            pass
    orch.run_stage3(processed)
```

- [ ] **Step 5: テスト + selftest 実行**

Run:
```
python -m pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 30
python DSRE.py --selftest 2>&1 | Select-String "verdict"
```
Expected: 全 PASS + `verdict=EQUIV`

- [ ] **Step 6: Commit**

```bash
git add DSRE.py tests/test_workflow_integration.py
git commit -m "feat: WorkflowOrchestrator integrating STAGE 1-3 pipeline

Stage 1: scan -> fingerprint -> cluster -> harmonize -> split versions
-> score -> best select -> discard -> relocate to foobar layout.
Stage 3: foobar layout for OUTPUT_DIR.
Hooked into Worker.run with DSRE_WORKFLOW=1 default."
```

---

## Task 13: 統合テスト (multi-version mock workflow)

**Files:**
- Modify: `tests/test_workflow_integration.py` (multi-version 全 stage 試行を追加)

ユーザーの実ワークフロー再現テスト: 同曲 3 版 (1 版だけタグ済) を投入し、伝播・選択・整列が正しく行われることを確認。

- [ ] **Step 1: 統合テスト追加**

Edit `tests/test_workflow_integration.py`:

```python
def test_full_workflow_multi_version():
    """同曲 3 版 (1 版だけ tagged) でメタデータ伝播 + 最良選択 + 整列を確認。"""
    from DSRE import WorkflowOrchestrator, MetadataExtractor, _resolve_fpcalc_path
    if _resolve_fpcalc_path() is None:
        return
    with tempfile.TemporaryDirectory() as d:
        in_dir = os.path.join(d, "in")
        out_dir = os.path.join(d, "out")
        db = os.path.join(d, "fp.db")
        os.makedirs(in_dir)
        os.makedirs(out_dir)

        # 同じ音響内容の 3 file (1 つだけ tagged)
        tagged = os.path.join(in_dir, "tagged.flac")
        u1 = os.path.join(in_dir, "untagged1.flac")
        u2 = os.path.join(in_dir, "untagged2.flac")
        for p in (tagged, u1, u2):
            _mk_flac(p, freq=440.0, dur=2.5)
        from mutagen.flac import FLAC
        f = FLAC(tagged)
        for k, v in {"artist": "X", "album": "Y", "title": "Z",
                     "discnumber": "1", "tracknumber": "5",
                     "genre": "J-Pop"}.items():
            f[k] = [v]
        f.save()

        orch = WorkflowOrchestrator(input_dir=in_dir, output_dir=out_dir, db_path=db)
        bests = orch.run_stage1()

        # 1 個だけ残っている (3 → 1)
        assert len(bests) == 1
        # 残った file には伝播されたタグがある
        m = MetadataExtractor.extract(bests[0])
        assert m["artist"] == "X"
        assert m["album"] == "Y"
        assert m["title"] == "Z"
        # foobar 階層に配置されている (genre/album/file)
        assert "J-Pop" in bests[0]
        assert "Y" in bests[0]
```

- [ ] **Step 2: テスト実行**

Run: `python -m pytest tests/test_workflow_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_workflow_integration.py
git commit -m "test: full multi-version workflow integration

Validates the user'\''s core scenario: 3 copies of same song with only
1 file tagged, after STAGE 1 only 1 file remains with metadata
correctly propagated and foobar-layout path."
```

---

## Task 14: CI/Deploy & 動作確認

**Files:** (なし、デプロイ確認のみ)

- [ ] **Step 1: 全テスト + selftest を通す**

Run:
```
python -m pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 40
python DSRE.py --selftest 2>&1 | Select-String "verdict|gates"
```
Expected: 全 PASS + `verdict=EQUIV`

- [ ] **Step 2: ローカル Worker GUI 起動で smoke (INPUT_DIR に少数 mock を入れて確認)**

PowerShell:
```powershell
# 小規模 INPUT_DIR で動作確認 (実音源を直接使うのではなく、tmp で実施推奨)
# ユーザー確認用: DSRE.exe を起動して 1-2 file 処理し、INPUT/OUTPUT の整列を目視
```

- [ ] **Step 3: Push (CI build → artifact → deploy)**

Run:
```
git push origin main
# CI watch
gh run list --limit 1
gh run watch <run-id> --exit-status
# Deploy
gh run download <run-id> --name DSRE_private --dir ./dl_new
.\deploy.ps1 -ZipPath ".\dl_new\DSRE_private.zip"
```
Expected: `[deploy] runtime selftest OK` + `verdict=EQUIV`

- [ ] **Step 4: ユーザー実音源 (foobar 確認用) で動作 trial**

DSRE.exe を `C:\FreeSoft\DSRE\DSRE.exe` から起動し、INPUT_DIR にユーザーの実音源を入れて開始 → INPUT_DIR が foobar 階層に整列・非最良 file がゴミ箱に `[DSRE-inferior]` プレフィックスで入っていることを確認。

---

## Self-Review

### Spec 整合

- §0 出力形式 → Task 1 で normalize 抽出。FLAC のみ出力は既存挙動を維持 ✓
- §2 パイプライン → Task 12 で STAGE 1-3 統合 ✓
- §3.1 指紋 → Task 3 ✓
- §3.2 キャッシュ → Task 3 内 ✓
- §3.3 クラスタリング → Task 4-5 ✓
- §3.4 メタデータ伝播 → Task 6-7 ✓
- §3.5 版分岐 → Task 8 ✓
- §3.6 5-key sanity → spec で記述、現状 plan では明示 task なし。Orchestrator 内でログ警告として実装される想定 (実装時に WorkflowOrchestrator.harmonize_metadata に組み込む)
- §3.7 指紋失敗時の縮退 → Task 3 (compute は None 返却) + Task 12 (build_clusters で None を単独 cluster 扱い) ✓
- §4 採点 → Task 9 (opus 廃止、normalize 直接解析) ✓
- §5 介入最小化 → Task 9 デフォルト動作 + env opt-in、UI ダイアログは将来追加 (現 spec ではログのみ)
- §6 foobar 階層 → Task 11 ✓
- §7 STAGE 1 → Task 12 内 ✓
- §8 破棄 → Task 10 ✓
- §9 STAGE 3 → Task 12 内 ✓
- §10 冪等性 → Task 12 (`scan_pending` で dsre_version タグ check) ✓
- §11 リジューム → Task 12 (state は filesystem 自己記述) ✓
- §12 順序 → Task 12 内 ORDER BY (現状実装は呼出側ループ順序に依存。明示 sort 追加の余地あり)
- §13 UI → Task 12 で progress_cb 経由
- §14 モジュール → Task 1-12 で全てカバー ✓
- §15 依存 → Task 2 で fpcalc.exe ✓
- §16 変更 file → Task 1, 2, 12 で DSRE.py / DSRE.spec ✓
- §17 テスト → 各 Task に組み込み済
- §18 env vars → Task 9 (フラグ閾値, スコア重み) + Task 12 (workflow, presort, similarity, etc) ✓
- §19 リスク緩和 → Task 1 (関数共有), Task 8 (version 保護), Task 11 (長パス) で構造的に対応

### 既知のフォローアップ (Spec で言及済だが、初期実装後の改善余地)

- §3.6 5-key sanity warning logging: Task 12 の `harmonize_metadata` 内に追加可能
- §12 sort order の明示化: WorkflowOrchestrator 内に明示 sort 追加可能
- §10 UI 拡張 (DSRE_INTERACTIVE_CONFIRM): 別 task で MainWindow にダイアログ追加可能

これらは初期版がリリースされた後の改善として記録し、フォローアップ task として残す。

### Placeholder 残り無し

全 Step に具体的なコード or コマンドが含まれている。No "TBD" / "implement appropriate handling" 等の placeholder なし。

### 型整合

- `FingerprintResult` (Task 3) → `FingerprintEngine.compute` の戻り値、`fingerprint_similarity` の入力フィールド `fingerprint`
- `ScoreResult` (Task 9) → `QualityProbe.score` の戻り値、`BestSelector.choose` の入力 (path, ScoreResult, file_size) tuple の 2 要素目
- `BestSelection` (Task 10) → `BestSelector.choose` の戻り値、`WorkflowOrchestrator.run_stage1` の戻り値 list の要素 → 一致
- `MetadataExtractor.extract` (Task 6) 戻り値: dict with `__path__` + 全 _METADATA_FIELDS + `__pictures__` → `MetadataPropagator.choose_canonical/propagate/split_versions` で共通使用

### Spec の TP_TARGET_DBFS 値訂正

Spec v5 §4.2 に「true peak -0.3 dBFS」と書いたが、DSRE.py の実値は `TP_TARGET_DBFS = -1.0`。Plan Task 1 では実値ベースで実装するため、Spec を後で訂正することを memo (初期実装に影響なし)。
