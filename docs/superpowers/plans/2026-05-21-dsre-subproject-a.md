# DSRE Sub-project A 実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** アートワーク (JXL含む) 保持 / 処理メタデータ・品質メトリクス FLAC タグ埋め込み / グラフタブ追加

**Architecture:** `mutagen` でアートワーク抽出・再注入、MetricsComputer 結果を FLAC Vorbis Comment に書込、`pyqtgraph` ベースの MetricsTab を MainWindow のタブとして追加。メトリクス計算は1回のみ (FLAC タグ + SQLite で共有)。

**Tech Stack:** Python 3.11, PySide6, mutagen>=1.47, pyqtgraph>=0.13.7, sqlite3 (既存), MetricsComputer (既存)

---

## ファイル構成

| ファイル | 変更 |
|---|---|
| `DSRE.py` | 修正: `_extract_flac_pictures`, `_embed_output_metadata` 追加; `AudioMetricsLogger.log` 修正; `_process_one` 修正; `MetricsTab` クラス追加; `MainWindow.__init__` 修正 |
| `requirements.txt` | 修正: `mutagen>=1.47.0`, `pyqtgraph>=0.13.7` 追加 |
| `DSRE.spec` | 修正: `mutagen`, `pyqtgraph` を collect_all に追加 |
| `tests/test_artwork.py` | 新規: アートワーク保持テスト |
| `tests/test_metrics_embed.py` | 新規: メタデータ埋め込みテスト |
| `tests/test_metrics_tab.py` | 新規: MetricsTab インスタンス化テスト |

---

## Task 1: 依存関係追加

**Files:**
- Modify: `requirements.txt`
- Modify: `DSRE.spec`

- [ ] **Step 1: requirements.txt に追記**

```
mutagen>=1.47.0
pyqtgraph>=0.13.7
```

`requirements.txt` の最終行に追加する。ファイル全体:

```
PySide6==6.9.1
numpy==2.0.2
scipy==1.13.1
librosa==0.11.0
resampy==0.4.3
soundfile==0.13.1
send2trash==1.8.3
pyloudnorm==0.1.1
threadpoolctl==3.5.0
mutagen>=1.47.0
pyqtgraph>=0.13.7
```

- [ ] **Step 2: パッケージインストール**

```bash
pip install "mutagen>=1.47.0" "pyqtgraph>=0.13.7"
```

Expected: Successfully installed (既にある場合は Requirement already satisfied)

- [ ] **Step 3: DSRE.spec の collect_all ループに追加**

`DSRE.spec` の `for mod in (` ブロックに `"mutagen"` と `"pyqtgraph"` を追加:

```python
for mod in (
    "numpy",
    "scipy",
    "librosa",
    "numba",
    "llvmlite",
    "resampy",
    "soundfile",
    "send2trash",
    "audioread",
    "pooch",
    "soxr",
    "joblib",
    "sklearn",
    "threadpoolctl",
    "lazy_loader",
    "msgpack",
    "decorator",
    "cffi",
    "pyloudnorm",
    "future",
    "mutagen",
    "pyqtgraph",
):
```

- [ ] **Step 4: コミット**

```bash
git add requirements.txt DSRE.spec
git commit -m "deps: add mutagen and pyqtgraph"
```

---

## Task 2: アートワーク抽出ヘルパー

**Files:**
- Modify: `DSRE.py` (AudioMetricsLogger クラスの直後、line ~673 付近に挿入)
- Create: `tests/test_artwork.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_artwork.py`:

```python
import os, sys, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"  # JXL signature bytes (最小フェイク)


def _make_test_flac(path: str) -> None:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3] * 2, axis=1).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def _embed_picture(flac_path: str, img_bytes: bytes, mime: str = "image/jxl") -> None:
    from mutagen.flac import FLAC, Picture
    f = FLAC(flac_path)
    pic = Picture()
    pic.type = 3  # Front cover
    pic.mime = mime
    pic.data = img_bytes
    f.add_picture(pic)
    f.save()


def test_extract_flac_pictures_returns_list():
    from DSRE import _extract_flac_pictures
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.flac")
        _make_test_flac(path)
        result = _extract_flac_pictures(path)
        assert isinstance(result, list)
        assert len(result) == 0  # アートワーク無しは空リスト


def test_extract_flac_pictures_captures_jxl():
    from DSRE import _extract_flac_pictures
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.flac")
        _make_test_flac(path)
        _embed_picture(path, _FAKE_JXL, mime="image/jxl")
        result = _extract_flac_pictures(path)
        assert len(result) == 1
        assert result[0].data == _FAKE_JXL
        assert result[0].mime == "image/jxl"


def test_extract_flac_pictures_nonexistent_returns_empty():
    from DSRE import _extract_flac_pictures
    result = _extract_flac_pictures("/nonexistent/path.flac")
    assert result == []
```

- [ ] **Step 2: テスト失敗を確認**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_artwork.py -v 2>&1 | head -20
```

Expected: ImportError または AttributeError (`_extract_flac_pictures` が存在しない)

- [ ] **Step 3: `_extract_flac_pictures` を DSRE.py に追加**

`DSRE.py` の `AudioMetricsLogger` クラス定義の後 (line ~673, `def _resource_base_dirs()` の直前) に追加:

```python
def _extract_flac_pictures(path: str) -> list:
    """FLAC ファイルから PICTURE ブロックを mutagen で抽出する。失敗時は空リストを返す。"""
    try:
        from mutagen.flac import FLAC
        return list(FLAC(path).pictures)
    except Exception:
        return []
```

- [ ] **Step 4: テスト実行**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_artwork.py::test_extract_flac_pictures_returns_list tests/test_artwork.py::test_extract_flac_pictures_captures_jxl tests/test_artwork.py::test_extract_flac_pictures_nonexistent_returns_empty -v
```

Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add DSRE.py tests/test_artwork.py
git commit -m "feat: add _extract_flac_pictures helper"
```

---

## Task 3: メタデータ埋め込みヘルパー

**Files:**
- Modify: `DSRE.py` (`_extract_flac_pictures` の直後に追加)
- Create: `tests/test_metrics_embed.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_metrics_embed.py`:

```python
import os, sys, tempfile, datetime
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SR = 96000
_FAKE_JXL = b"\xff\x0a\x00\x00\x00\x00\x00\x00"


def _make_test_flac(path: str) -> None:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    audio = np.stack([np.sin(2 * np.pi * 440 * t) * 0.3] * 2, axis=1).astype(np.float32)
    sf.write(path, audio, SR, subtype="PCM_24", format="FLAC")


def _make_before_after():
    from DSRE import MetricsComputer
    t = np.linspace(0, 1.0, SR, endpoint=False)
    mono = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    before = MetricsComputer.compute(mono, SR)
    after = MetricsComputer.compute(mono * 1.01, SR)
    return before, after


def test_embed_output_metadata_adds_dsre_version():
    from DSRE import _embed_output_metadata
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, [], before, after, level=5)
        f = FLAC(path)
        assert "dsre_version" in f
        assert "dsre_processed_utc" in f
        assert "dsre_level" in f
        assert f["dsre_level"][0] == "5"


def test_embed_output_metadata_adds_before_after_metrics():
    from DSRE import _embed_output_metadata
    from mutagen.flac import FLAC
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, [], before, after, level=5)
        f = FLAC(path)
        assert "dsre_before_rms_db" in f
        assert "dsre_after_rms_db" in f
        assert "dsre_before_dr" in f
        assert "dsre_after_dr" in f


def test_embed_output_metadata_preserves_pictures():
    from DSRE import _embed_output_metadata, _extract_flac_pictures
    from mutagen.flac import FLAC, Picture
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        # アートワーク付き FLAC を作る
        pic = Picture()
        pic.type = 3
        pic.mime = "image/jxl"
        pic.data = _FAKE_JXL
        f = FLAC(path)
        f.add_picture(pic)
        f.save()
        pictures = _extract_flac_pictures(path)
        before, after = _make_before_after()
        _embed_output_metadata(path, pictures, before, after, level=5)
        result_pics = _extract_flac_pictures(path)
        assert len(result_pics) == 1
        assert result_pics[0].data == _FAKE_JXL
        assert result_pics[0].mime == "image/jxl"


def test_embed_output_metadata_no_crash_on_none_metrics():
    from DSRE import _embed_output_metadata
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.flac")
        _make_test_flac(path)
        # None 値を含む metrics でもクラッシュしない
        before = {k: None for k in ["rms_db", "peak_db", "dr", "plr", "lufs"]}
        after = {k: None for k in ["rms_db", "peak_db", "dr", "plr", "lufs"]}
        _embed_output_metadata(path, [], before, after, level=1)  # should not raise
```

- [ ] **Step 2: テスト失敗確認**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_metrics_embed.py -v 2>&1 | head -20
```

Expected: ImportError (`_embed_output_metadata` が存在しない)

- [ ] **Step 3: `_embed_output_metadata` を DSRE.py に追加**

`_extract_flac_pictures` の直後に追加:

```python
def _embed_output_metadata(
    path: str,
    pictures: list,
    before_m: dict,
    after_m: dict,
    level: int,
) -> None:
    """処理済 FLAC に PICTURE ブロック + カスタム Vorbis Comment タグを埋め込む。失敗は握り潰す。"""
    try:
        import datetime
        from mutagen.flac import FLAC
        f = FLAC(path)
        # PICTURE ブロック再注入 (元の順序で)
        f.clear_pictures()
        for pic in pictures:
            f.add_picture(pic)
        # カスタムタグ (小文字: FLAC Vorbis Comment 規約)
        if f.tags is None:
            f.add_tags()
        f.tags["dsre_version"] = [_get_dsre_version()]
        f.tags["dsre_processed_utc"] = [datetime.datetime.utcnow().isoformat()]
        f.tags["dsre_level"] = [str(level)]
        for k, v in before_m.items():
            if v is not None:
                f.tags[f"dsre_before_{k}"] = [str(round(float(v), 6))]
        for k, v in after_m.items():
            if v is not None:
                f.tags[f"dsre_after_{k}"] = [str(round(float(v), 6))]
        f.save()
    except Exception:
        pass  # メタデータ失敗は本処理をブロックしない
```

- [ ] **Step 4: テスト実行**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_metrics_embed.py -v
```

Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add DSRE.py tests/test_metrics_embed.py
git commit -m "feat: add _embed_output_metadata helper"
```

---

## Task 4: AudioMetricsLogger に pre-computed metrics サポート追加

**Files:**
- Modify: `DSRE.py` (AudioMetricsLogger.log メソッド)

- [ ] **Step 1: `AudioMetricsLogger.log` の signature と実装を変更**

現在の `log` メソッド signature:

```python
@classmethod
def log(
    cls,
    input_path: str,
    input_audio,
    output_audio,
    sr: int,
    processing_time_sec: float,
    adaptive_layers=None,
    adaptive_mode=None,
    hf_ratio_input=None,
) -> None:
```

変更後 (`_before_metrics` / `_after_metrics` を追加):

```python
@classmethod
def log(
    cls,
    input_path: str,
    input_audio,
    output_audio,
    sr: int,
    processing_time_sec: float,
    adaptive_layers=None,
    adaptive_mode=None,
    hf_ratio_input=None,
    _before_metrics: "dict | None" = None,
    _after_metrics: "dict | None" = None,
) -> None:
    """
    メトリクスを計算して DB に 1 行追記。例外は握り潰す（処理をブロックしない）。
    _before_metrics / _after_metrics: 事前計算済みの場合は MetricsComputer を再呼びしない。
    """
```

メソッド本体の `b = MetricsComputer.compute(input_audio, sr)` / `a = MetricsComputer.compute(output_audio, sr)` の 2 行を以下に置換:

```python
b = _before_metrics if _before_metrics is not None else MetricsComputer.compute(input_audio, sr)
a = _after_metrics if _after_metrics is not None else MetricsComputer.compute(output_audio, sr)
```

- [ ] **Step 2: 既存テストが通ることを確認 (後退回避)**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_metrics.py tests/test_metrics_embed.py -v
```

Expected: all passed

- [ ] **Step 3: コミット**

```bash
git add DSRE.py
git commit -m "feat: AudioMetricsLogger accepts pre-computed metrics to avoid double compute"
```

---

## Task 5: `_process_one` 修正 (アートワーク抽出・メタデータ埋め込み統合)

**Files:**
- Modify: `DSRE.py` (`_process_one` 内部、Worker.run の内部関数)

`_process_one` の現在の構造:

```
ディスク確認
y, sr = load_audio_safe(path)
trim_silence
resample
zansei_impl → y_out
save_flac24_out
AudioMetricsLogger.log
send2trash
```

変更後の構造:

```
ディスク確認
y, sr = load_audio_safe(path)
pictures = _extract_flac_pictures(path)      ← NEW
trim_silence
resample
before_m = MetricsComputer.compute(y, sr)   ← NEW
zansei_impl → y_out
after_m = MetricsComputer.compute(y_out, sr) ← NEW
save_flac24_out
_embed_output_metadata(out, pictures, before_m, after_m, lv)  ← NEW
AudioMetricsLogger.log(..., _before_metrics=before_m, _after_metrics=after_m)  ← MODIFIED
send2trash
```

- [ ] **Step 1: `_check_abort()` 後、`y, sr = load_audio_safe(path)` の直後に追加**

`_process_one` 内 (line ~2009 付近) の `y, sr = load_audio_safe(path)` の直後に:

```python
pictures = _extract_flac_pictures(path)
```

- [ ] **Step 2: `zansei_impl` の呼び出し直前に before_m 計算を追加**

`t0 = time.perf_counter()` の直前に:

```python
before_m = MetricsComputer.compute(y, sr)
_check_abort()
```

- [ ] **Step 3: zansei_impl 呼び出し後、save_flac24_out の前に after_m 計算を追加**

`proc_time = time.perf_counter() - t0` の直後に:

```python
after_m = MetricsComputer.compute(y_out, sr)
```

- [ ] **Step 4: save_flac24_out 呼び出しの直後に _embed_output_metadata を追加**

`save_flac24_out(...)` の try/except ブロック終了後、`_check_abort()` の直前に:

```python
try:
    _embed_output_metadata(out, pictures, before_m, after_m, lv)
except Exception:
    pass  # メタデータ失敗は処理を止めない
```

- [ ] **Step 5: AudioMetricsLogger.log の呼び出しを修正**

既存の呼び出し:

```python
AudioMetricsLogger.log(
    input_path=path,
    input_audio=y,
    output_audio=y_out,
    sr=sr,
    processing_time_sec=proc_time,
)
```

変更後:

```python
AudioMetricsLogger.log(
    input_path=path,
    input_audio=y,
    output_audio=y_out,
    sr=sr,
    processing_time_sec=proc_time,
    _before_metrics=before_m,
    _after_metrics=after_m,
)
```

- [ ] **Step 6: selftest 実行**

```bash
cd C:\Users\reinb\src\DSRE && python DSRE.py --selftest
```

Expected: verdict ≠ DEGRADED、すべての 5 層テスト PASS

- [ ] **Step 7: コミット**

```bash
git add DSRE.py
git commit -m "feat: preserve artwork and embed processing metadata in output FLAC"
```

---

## Task 6: MetricsTab ウィジェット実装

**Files:**
- Modify: `DSRE.py` (`MainWindow` クラスの直前に追加)
- Create: `tests/test_metrics_tab.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_metrics_tab.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_metrics_tab_instantiates():
    """MetricsTab が Qt アプリ内でクラッシュなしにインスタンス化できること。"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from DSRE import MetricsTab
    tab = MetricsTab()
    assert tab is not None


def test_metrics_tab_refresh_no_crash():
    """DB が空でも refresh() がクラッシュしないこと。"""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from DSRE import MetricsTab
    tab = MetricsTab()
    tab.refresh()  # should not raise
```

- [ ] **Step 2: テスト失敗確認**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_metrics_tab.py -v 2>&1 | head -10
```

Expected: ImportError (MetricsTab が存在しない)

- [ ] **Step 3: MetricsTab クラスを DSRE.py に追加**

`class _NullCtx` (line ~1824) の直前に以下を挿入:

```python
class MetricsTab(QtWidgets.QWidget):
    """SQLite から音響メトリクスを読み込んで pyqtgraph で可視化するタブ。"""

    _METRIC_LABELS = [
        ("rms_db", "RMS (dBFS)"),
        ("peak_db", "Peak (dBFS)"),
        ("dr", "DR"),
        ("plr", "PLR"),
        ("lufs", "LUFS"),
        ("lra", "LRA"),
        ("clip_count", "Clip Count"),
        ("centroid_hz", "Centroid (Hz)"),
        ("rolloff_hz", "Rolloff (Hz)"),
        ("flatness", "Flatness"),
        ("hf_ratio_4k", "HF ratio >4kHz"),
        ("hf_ratio_8k", "HF ratio >8kHz"),
        ("hf_ratio_12k", "HF ratio >12kHz"),
        ("hf_ratio_16k", "HF ratio >16kHz"),
        ("harmonic_1k_proxy", "Harmonic proxy"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            import pyqtgraph as pg
            pg.setConfigOption("background", "#1e1e1e")
            pg.setConfigOption("foreground", "#cccccc")
            self._pg = pg
        except ImportError:
            self._pg = None

        self._sub = QtWidgets.QTabWidget()

        # --- 最新バッチタブ ---
        batch_container = QtWidgets.QWidget()
        batch_layout = QtWidgets.QVBoxLayout(batch_container)
        self._metric_combo_batch = QtWidgets.QComboBox()
        for key, label in self._METRIC_LABELS:
            self._metric_combo_batch.addItem(label, key)
        batch_layout.addWidget(self._metric_combo_batch)
        if self._pg:
            self._batch_plot = self._pg.PlotWidget()
            batch_layout.addWidget(self._batch_plot)
        else:
            self._batch_plot = None
            batch_layout.addWidget(QtWidgets.QLabel("pyqtgraph が必要です"))
        self._sub.addTab(batch_container, "最新バッチ")

        # --- 全履歴タブ ---
        hist_container = QtWidgets.QWidget()
        hist_layout = QtWidgets.QVBoxLayout(hist_container)
        self._metric_combo_hist = QtWidgets.QComboBox()
        for key, label in self._METRIC_LABELS:
            self._metric_combo_hist.addItem(label, key)
        hist_layout.addWidget(self._metric_combo_hist)
        if self._pg:
            self._hist_plot = self._pg.PlotWidget()
            hist_layout.addWidget(self._hist_plot)
        else:
            self._hist_plot = None
            hist_layout.addWidget(QtWidgets.QLabel("pyqtgraph が必要です"))
        self._sub.addTab(hist_container, "全履歴")

        btn_refresh = QtWidgets.QPushButton("更新")
        btn_refresh.clicked.connect(self.refresh)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._sub)
        layout.addWidget(btn_refresh)

        self._metric_combo_batch.currentIndexChanged.connect(self._draw_batch)
        self._metric_combo_hist.currentIndexChanged.connect(self._draw_hist)

    def refresh(self) -> None:
        self._draw_batch()
        self._draw_hist()

    def _load_batch(self, n: int = 50) -> list[dict]:
        """直近 n 件のレコードを SQLite から取得。失敗時は空リスト。"""
        try:
            with sqlite3.connect(METRICS_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,)
                ).fetchall()
                return [dict(r) for r in reversed(rows)]
        except Exception:
            return []

    def _load_history(self, n: int = 500) -> list[dict]:
        """全履歴 (最大 n 件) を取得。"""
        try:
            with sqlite3.connect(METRICS_DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY id ASC LIMIT ?", (n,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def _draw_batch(self) -> None:
        if not self._pg or self._batch_plot is None:
            return
        key = self._metric_combo_batch.currentData()
        if not key:
            return
        rows = self._load_batch()
        before_key = f"{key}_b"
        after_key = f"{key}_a"
        before_vals = [r.get(before_key) for r in rows if r.get(before_key) is not None]
        after_vals = [r.get(after_key) for r in rows if r.get(after_key) is not None]
        n = min(len(before_vals), len(after_vals))
        if n == 0:
            self._batch_plot.clear()
            return
        x = list(range(n))
        bar_width = 0.35
        self._batch_plot.clear()
        bg = self._pg.BarGraphItem(
            x=[xi - bar_width / 2 for xi in x],
            height=before_vals[:n],
            width=bar_width,
            brush="#607D8B",
            name="Before",
        )
        ba = self._pg.BarGraphItem(
            x=[xi + bar_width / 2 for xi in x],
            height=after_vals[:n],
            width=bar_width,
            brush="#4CAF50",
            name="After",
        )
        self._batch_plot.addItem(bg)
        self._batch_plot.addItem(ba)
        self._batch_plot.setLabel("bottom", "File index")
        self._batch_plot.setLabel("left", self._metric_combo_batch.currentText())

    def _draw_hist(self) -> None:
        if not self._pg or self._hist_plot is None:
            return
        key = self._metric_combo_hist.currentData()
        if not key:
            return
        rows = self._load_history()
        before_key = f"{key}_b"
        after_key = f"{key}_a"
        before_vals = [r.get(before_key) for r in rows]
        after_vals = [r.get(after_key) for r in rows]
        xs = list(range(len(rows)))
        self._hist_plot.clear()
        b_clean = [(i, v) for i, v in zip(xs, before_vals) if v is not None]
        a_clean = [(i, v) for i, v in zip(xs, after_vals) if v is not None]
        if b_clean:
            bx, by = zip(*b_clean)
            self._hist_plot.plot(list(bx), list(by), pen="#607D8B", name="Before")
        if a_clean:
            ax, ay = zip(*a_clean)
            self._hist_plot.plot(list(ax), list(ay), pen="#4CAF50", name="After")
        self._hist_plot.setLabel("bottom", "Run index")
        self._hist_plot.setLabel("left", self._metric_combo_hist.currentText())
        legend = self._hist_plot.addLegend()
        legend.setOffset((10, 10))
```

- [ ] **Step 4: テスト実行**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_metrics_tab.py -v
```

Expected: 2 passed

- [ ] **Step 5: コミット**

```bash
git add DSRE.py tests/test_metrics_tab.py
git commit -m "feat: add MetricsTab widget with before/after and history views"
```

---

## Task 7: MainWindow にタブ構造を追加

**Files:**
- Modify: `DSRE.py` (`MainWindow.__init__`, line ~2186)

現在の MainWindow は全ウィジェットを直接 QVBoxLayout に入れている。これを QTabWidget でラップし「処理」「グラフ」の 2 タブにする。

- [ ] **Step 1: MainWindow.__init__ を修正**

現在の `__init__` の `layout = QtWidgets.QVBoxLayout()` から `self.setLayout(layout)` までを以下に置換:

```python
# ---- 処理タブ (既存UIを QWidget でラップ) ----
proc_widget = QtWidgets.QWidget()
proc_layout = QtWidgets.QVBoxLayout(proc_widget)
proc_layout.addWidget(self.label)
proc_layout.addWidget(self.pb_file)
proc_layout.addWidget(self.pb_all)
proc_layout.addWidget(self.btn_start)
proc_layout.addWidget(self.btn_pause)
proc_layout.addWidget(self.btn_cancel)
row = QtWidgets.QHBoxLayout()
row.addWidget(self.lbl_level)
row.addWidget(self.sld_level, 1)
proc_layout.addLayout(row)

# ---- グラフタブ ----
self.metrics_tab = MetricsTab()

# ---- タブウィジェット ----
self._tabs = QtWidgets.QTabWidget()
self._tabs.addTab(proc_widget, "処理")
self._tabs.addTab(self.metrics_tab, "グラフ")

main_layout = QtWidgets.QVBoxLayout()
main_layout.setContentsMargins(0, 0, 0, 0)
main_layout.addWidget(self._tabs)
self.setLayout(main_layout)

self.resize(400, 280)
```

- [ ] **Step 2: Worker 完了シグナルを MetricsTab.refresh に接続**

`MainWindow.start` メソッド内で Worker を開始する箇所 (Worker のシグナル接続後) に以下を追加:

まず `start` メソッドを探し (`def start(self, files=None):`)、`self.worker.finished.connect(...)` または同等のシグナル接続の後に:

```python
self.worker.finished.connect(self.metrics_tab.refresh)
```

- [ ] **Step 3: 動作確認 (手動)**

```bash
cd C:\Users\reinb\src\DSRE && python DSRE.py
```

Expected: ウィンドウが「処理」「グラフ」のタブ付きで開く。クラッシュなし。

- [ ] **Step 4: selftest**

```bash
cd C:\Users\reinb\src\DSRE && python DSRE.py --selftest
```

Expected: verdict ≠ DEGRADED

- [ ] **Step 5: 全テスト実行**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/ -v
```

Expected: all passed

- [ ] **Step 6: コミット**

```bash
git add DSRE.py
git commit -m "feat: add graph tab to MainWindow, auto-refresh after processing"
```

---

## Task 8: エンドツーエンド検証

**Files:** なし (既存ファイルのみ)

- [ ] **Step 1: アートワーク保持 統合テスト**

```bash
cd C:\Users\reinb\src\DSRE && python -m pytest tests/test_artwork.py tests/test_metrics_embed.py -v
```

Expected: all passed

- [ ] **Step 2: selftest 最終確認**

```bash
cd C:\Users\reinb\src\DSRE && python DSRE.py --selftest
```

Expected: 5 層すべて PASS、verdict ≠ DEGRADED

- [ ] **Step 3: CI push & 監視**

```bash
git push origin main
```

Expected: GitHub Actions が起動し 270 秒後に確認 (`gh run watch`)

- [ ] **Step 4: deploy**

CI 成功後:

```bash
gh run download <run_id> --name DSRE_private --dir ./dl_tmp_v_meta
.\deploy.ps1 -ZipPath ".\dl_tmp_v_meta\DSRE_private.zip"
```

- [ ] **Step 5: 実音源での動作確認**

JXL アートワーク付き FLAC を DSRE で処理し:
1. 出力 FLAC を mp3tag で開いてアートワーク表示確認
2. 出力 FLAC を mutagen で確認: `DSRE_VERSION`, `DSRE_BEFORE_DR`, `DSRE_AFTER_HF_RATIO_12K` 等が存在
3. グラフタブで「更新」→ Before/After グラフが表示される

---

## 補足: Sub-project B/C/D 実装時の前提

**Sub-project B (重複検出・フォルダ仕分け)** は Sub-project A の MetricsComputer 結果を前提とする。A 完了後に着手。

**Sub-project C (メタデータ自動取得)** は別ツール。VGMdb API: `https://vgmdb.info/` (JSON)、MusicBrainz: `https://musicbrainz.org/ws/2/`。DSRE には組み込まない。

**Sub-project D (主観品質フラグ)** は A の clip_count / flatness / harmonic_1k_proxy を使い、処理後に「要確認」UI フラグを立てる。A 完了後に検討。
