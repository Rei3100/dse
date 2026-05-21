# DSRE Sub-project A: コア改善 設計書

**Date:** 2026-05-21  
**Scope:** アートワーク保持 / メタデータ埋め込み / 品質グラフタブ  
**Out of scope (Sub-project B):** 重複検出・フォルダ自動仕分け・処理順序メタデータ基準  
**Out of scope (C/D, 将来):** メタデータ自動取得・主観品質ゲート  

---

## 1. アートワーク保持 (Bug Fix)

### 問題

`save_flac24_out` の ffmpeg コマンド (`-map 0:a -map_metadata 1 -c copy`) は Vorbis Comments を継承するが、FLAC PICTURE ブロック (アートワーク) は `-map 0:a` により除外される。JXL を含む全フォーマットに影響。

### 解決策

`mutagen.flac.FLAC` を使い PICTURE ブロックを生バイトで抽出・再埋め込み。ffmpeg を介さないため JXL を含む任意のフォーマットに対応。

### 実装

**新規関数 2つ (DSRE.py 内):**

```python
def _extract_flac_pictures(path: str) -> list:
    """入力 FLAC の PICTURE ブロックを mutagen で抽出して返す。失敗時は空リスト。"""
    from mutagen.flac import FLAC
    try:
        return list(FLAC(path).pictures)
    except Exception:
        return []

def _embed_output_metadata(
    path: str,
    pictures: list,
    before_m: dict,
    after_m: dict,
    level: int,
) -> None:
    """処理済 FLAC に PICTURE + カスタム Vorbis Comment タグを埋め込む。失敗は握り潰す。"""
    from mutagen.flac import FLAC
    import datetime
    try:
        f = FLAC(path)
        # PICTURE ブロック再注入
        f.clear_pictures()
        for pic in pictures:
            f.add_picture(pic)
        # カスタムタグ
        tags = {
            "DSRE_VERSION": [_get_dsre_version()],
            "DSRE_PROCESSED_UTC": [datetime.datetime.utcnow().isoformat()],
            "DSRE_LEVEL": [str(level)],
        }
        for k, v in before_m.items():
            if v is not None:
                tags[f"DSRE_BEFORE_{k.upper()}"] = [str(round(v, 6))]
        for k, v in after_m.items():
            if v is not None:
                tags[f"DSRE_AFTER_{k.upper()}"] = [str(round(v, 6))]
        if f.tags is None:
            f.add_tags()
        f.tags.update(tags)
        f.save()
    except Exception:
        pass  # メタデータ失敗は本処理をブロックしない
```

**`_process_one` の変更点:**

```python
# 既存: y, sr = load_audio_safe(path) の直後に追加
pictures = _extract_flac_pictures(path)

# 既存: zansei_impl の後、save_flac24_out の前に追加
before_m = MetricsComputer.compute(y, sr)

# 既存: save_flac24_out の後に追加
after_m = MetricsComputer.compute(y_out, sr)
_embed_output_metadata(out, pictures, before_m, after_m, lv)

# 既存: AudioMetricsLogger.log に pre-computed metrics を渡す (再計算不要)
AudioMetricsLogger.log(
    input_path=path,
    input_audio=y,
    output_audio=y_out,
    sr=sr,
    processing_time_sec=proc_time,
    _before_metrics=before_m,   # 新パラメータ
    _after_metrics=after_m,     # 新パラメータ
)
```

**`AudioMetricsLogger.log` の変更点:**

`_before_metrics` / `_after_metrics` が `None` なら従来通り `MetricsComputer.compute()` を呼ぶ。提供された場合はスキップ (二重計算防止)。

---

## 2. 処理メタデータ埋め込み

`_embed_output_metadata` が埋め込む FLAC Vorbis Comment タグ一覧:

| タグ名 | 値例 | 説明 |
|---|---|---|
| `DSRE_VERSION` | `2.1.0` | 処理時の DSRE バージョン |
| `DSRE_PROCESSED_UTC` | `2026-05-21T14:30:00` | 処理日時 (UTC ISO 8601) |
| `DSRE_LEVEL` | `5` | 処理レベル (1-10) |
| `DSRE_BEFORE_RMS_DB` | `-14.23` | 処理前 RMS |
| `DSRE_AFTER_RMS_DB` | `-14.19` | 処理後 RMS |
| `DSRE_BEFORE_DR` | `12.4` | 処理前 Dynamic Range |
| `DSRE_AFTER_DR` | `12.4` | 処理後 Dynamic Range |
| `DSRE_BEFORE_LUFS` | `-16.1` | 処理前 LUFS |
| `DSRE_AFTER_LUFS` | `-16.0` | 処理後 LUFS |
| `DSRE_BEFORE_HF_RATIO_12K` | `0.023` | 処理前 12kHz 以上エネルギー比 |
| `DSRE_AFTER_HF_RATIO_12K` | `0.041` | 処理後 12kHz 以上エネルギー比 |
| … (全 17 メトリクス × before/after) | | MetricsComputer の全出力 |

---

## 3. 品質グラフタブ

### UI 構成

MainWindow に `QTabWidget` を追加し「処理」「グラフ」の 2 タブ構成にする。

```
MainWindow
├── QTabWidget
│   ├── Tab "処理" (既存の処理UI)
│   └── Tab "グラフ" (新規 MetricsTab)
│       ├── QTabWidget (サブタブ)
│       │   ├── "最新バッチ" - Before/After 棒グラフ
│       │   └── "全履歴" - 時系列折れ線グラフ
│       └── 更新ボタン
```

### MetricsTab クラス

```python
class MetricsTab(QtWidgets.QWidget):
    """SQLite から音響メトリクスを読み込んで可視化する。"""
    
    def __init__(self, parent=None): ...
    def refresh(self) -> None: ...    # Worker 完了シグナルにも接続
    def _load_recent(self) -> list:   # 最新バッチ (直近 N ファイル) をDBから取得
    def _load_history(self) -> list:  # 全履歴をDBから取得
    def _build_batch_view(self): ...  # Before/After 棒グラフ (pyqtgraph)
    def _build_history_view(self): ... # 時系列 (pyqtgraph PlotWidget)
```

### 最新バッチビュー

- x軸: ファイル名 (短縮)
- 2色棒グラフ: Before (灰) / After (青緑)
- 表示メトリクス切替ドロップダウン (全17選択可)
- 差分 Δ を棒上に数値表示

### 全履歴ビュー

- x軸: 処理日時
- y軸: 選択メトリクス値
- Before/After 両折れ線を重ねて表示
- ファイル数・処理日時範囲のフィルタースライダー

### データソース

既存 `AudioMetricsLogger` の SQLite DB を直読み。DB パス: `_db_path()` 経由。

---

## 4. 依存関係追加

`requirements.txt` に追加:

```
mutagen>=1.47.0
pyqtgraph>=0.13.7
```

---

## 5. 変更ファイル

| ファイル | 変更種別 | 概要 |
|---|---|---|
| `DSRE.py` | 修正 | `_extract_flac_pictures`, `_embed_output_metadata` 追加; `_process_one` 修正; `AudioMetricsLogger.log` 修正; `MetricsTab` クラス追加; `MainWindow` にタブ追加 |
| `requirements.txt` | 修正 | `mutagen`, `pyqtgraph` 追加 |
| `DSRE.spec` | 修正 | `mutagen`, `pyqtgraph` の hidden imports / datas 追加 |

---

## 6. テスト / 検証

| 検証項目 | 方法 | 合格基準 |
|---|---|---|
| アートワーク保持 | JXL 入りサンプル FLAC を処理、mutagen で出力確認 | `len(FLAC(out).pictures) > 0` かつ bytes 一致 |
| JPG/PNG アートワーク | 同上 | 同上 |
| メタデータタグ | mutagen で出力 FLAC を開く | `DSRE_VERSION`, `DSRE_PROCESSED_UTC` 存在 |
| selftest | `python DSRE.py --selftest` | verdict ≠ DEGRADED |
| グラフタブ | 処理後にタブを開く | クラッシュなし・数値表示 |
| アートワーク無し | アートワーク無し FLAC を処理 | 例外なし・出力正常 |

---

## 7. 将来ドキュメント (Sub-project B/C/D)

### Sub-project B: ワークフロー自動化 (次セッション)

- **重複検出**: 同名または同メタデータのファイルを処理前にスキャン
- **品質ベース自動選別**: DR・LUFS・HF ratio を基準に最良 1 ファイルを自動選択、残りを `trash/` へ
- **フォルダ自動仕分け**: foobar2000 フォーマット互換のメタデータベースパス生成
  - 参考フォーマット: `Audio/{genre}/{age}/{circle}/{album}/{disc}/{track}`
  - ユーザーの foobar 式に準拠: アーティスト・サークル・ブランド・シリーズ階層
- **ファイル命名**: `{discnum}.{tracknum}.{title} - {artist} [feat.] [Prod.] [arrange] [version]`
- **処理キュー順序**: メタデータのアーティスト・サークル・年・アルバムでソート

### Sub-project C: メタデータ自動取得ツール (将来、別ツール推奨)

**課題**:
- VGMdb: アーティスト = CV声優名 → 使用時は artist タグをオフ
- MusicBrainz: アーティスト = キャラ名 → 使用時は artist タグをオン
- Apple Music / iTunes: 一般楽曲向け
- 設定が異なるため完全自動化は困難、人間確認ステップが残る

**理想ワークフロー**:
1. 音源フォルダをドロップ
2. ファイル名・既存メタデータから候補を自動検索 (VGMdb / MusicBrainz)
3. 候補をリスト表示、人間が選択・確認
4. 一括適用

**DSRE への組み込みより別ツールが適切な理由**: メタデータ取得は音響処理と無関係。API 管理・GUI が肥大化する。

### Sub-project D: 主観品質ゲート (将来、要検討)

**問題**: 「数値良好でも実際は劣化している音源」が存在する (過去実例あり)。数値: DR 高・clip 無し・HF ratio 良好 → しかし実聴で明らかに劣化。

**原因仮説**:
- 元音源が既にアーティファクト持ち (過圧縮・マスタリング歪み)
- DSRE がアーティファクトも倍音として増幅

**自動判定の限界**: 最終品質判断は聴感確認が必要。完全自動化は不可能。

**実装可能な部分**:
- 入力時に「要注意フラグ」を付ける: clip_count > 0 OR flatness < 閾値 OR THD proxy > 閾値
- フラグ付きファイルは処理完了後に UI 上で警告色表示
- ユーザーが手動確認する確率を減らす (完全ゼロは不可)
