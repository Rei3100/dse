# DSRE Sub-project B+D: ワークフロー自動化 + 客観品質ゲート 設計書

**Date:** 2026-05-21
**Scope:** 同曲グルーピング / 客観品質スコアリング / 自動最良選択 / opus 出力 / foobar 階層仕分け
**Integrated:** B (ワークフロー) と D (品質ゲート) を統合。品質判定は「自動選別の入力データ」として B に組み込まれる
**Premise:** 全自動運用がデフォルト。要注意フラグ時のみユーザー確認

---

## 0. 目的 (ワンライン)

INPUT_DIR の楽曲群を解析・グルーピングし、同曲の複数バージョンから客観品質最良の 1 つを選択 → DSRE 処理 → opus 出力 → foobar 階層へ自動仕分け。残りは送信トラッシュへ。

---

## 1. 全体パイプライン (DAG)

```
[INPUT_DIR scan]
       │
       ▼
[metadata extract] (mutagen)
       │
       ▼
[grouping] (artist + title + album → group id)
       │
       ▼
[quality analysis] (MetricsComputer per file)
       │
       ▼
[score + flag] (QualityScorer)
       │
       ▼
[best selection] (group 内 max score, フラグ判定)
       │
       ▼ (best のみ)
[DSRE processing] (zansei_impl, 既存)
       │
       ▼
[opus encode] (ffmpeg libopus 256k VBR)
       │
       ▼
[foobar path build] (template + sanitize)
       │
       ▼
[final placement] (OUTPUT_DIR/{path}.opus)
       │
       ▼
[trash discarded versions + intermediate FLAC]
```

---

## 2. 同曲グルーピング (重複検出)

### 2.1 判定ロジック (段階的)

**段階 1: メタデータ完全一致**
```python
key = (
    normalize(artist),
    normalize(title_strip_version_suffix),
    normalize(album),
)
```
`normalize`: lowercase + 全角半角統一 + 連続空白 → 単一 + 前後 trim。

**段階 2: タイトル一致 + duration 近似**
段階 1 でグループ化されなかったファイルを duration ±2 秒以内かつ title 近似一致でマージ。

**段階 3: ファイル名 fuzzy match (fallback)**
メタデータ欠落時の安全網。`rapidfuzz.fuzz.token_set_ratio > 90`。

### 2.2 別曲扱い suffix (グルーピング対象外)

title 末尾の以下パターンは「別バージョン = 別曲」として保持:
- `[Live]`, `[Acoustic]`, `[Remix]`, `[Instrumental]`, `[Off Vocal]`, `[Karaoke]`, `[TV Size]`
- `(Live)`, `(Acoustic)`, `(Remix)`, `(Instrumental)`, `(Off Vocal)`
- `-Live-`, `-Remix-`, `-Acoustic-`

サフィックス検出は `_VERSION_TAG_PATTERNS` 定数で正規表現管理。新パターン追加時はここを更新。

### 2.3 単独ファイル (グループサイズ 1)

そのまま処理対象。比較なしで DSRE → opus → 仕分けへ進む。

---

## 3. 客観品質スコアリング (B+D の核)

### 3.1 入力ファイルの解析

各 FLAC を読み込み、既存 `MetricsComputer.compute()` で 15 メトリクスを取得。重い処理 (1 ファイル数秒) なのでバッチで pre-cache する。

### 3.2 スコア式

0-100 のスケールに正規化。

```python
score = 0.0

# Dynamic Range (高いほど良好。max 60 点)
score += clamp(dr, 0, 20) * 3.0

# LUFS proximity to -14 (配信標準。max 20 点)
score += clamp(20 - abs(lufs + 14), 0, 20)

# 高域存在 (max 10 点)
score += clamp(hf_ratio_12k, 0, 0.05) * 200

# スペクトル平坦性 (高いほど自然。max 10 点)
score += clamp(flatness, 0, 0.5) * 20

# --- 減点 ---
score -= min(clip_count * 0.05, 20)           # クリップ過多 (max -20)
score -= max(0, peak_db + 0.1) * 5             # 0dBFS 張り付き (max -10)
score -= max(0, 6 - plr) * 2                   # PLR < 6 (ハイパー圧縮、max -12)

score = clamp(score, 0, 100)
```

期待値帯:
- 80-100: 高音質マスター (Hi-Res / 良好マスタリング)
- 60-80: 標準 CD 品質 (現代的マスタリング)
- 40-60: 過剰圧縮気味の現代 J-pop マスター
- 20-40: brick wall master, クリップ多発
- 0-20: 致命的劣化

### 3.3 要注意フラグ (主観品質ゲート D の自動化)

以下のいずれかに該当する場合「⚠ 要確認」フラグを立てる:

| フラグ条件 | 意味 |
|---|---|
| `clip_count > 100` | 大量クリップ。波形破壊の可能性 |
| `flatness < 0.05` | スペクトル平坦性極小 = brick wall |
| `hf_ratio_16k < 0.0005` | 高域カットされている (lossy 由来の可能性) |
| `plr < 4.0` | peak-to-loudness 危険水準 (ハイパー圧縮) |
| `dr < 6` | DR 危険水準 (聴感劣化リスク) |

### 3.4 自動選択ロジック

```python
def select_best(group: list[FileMetric]) -> Selection:
    no_flag = [f for f in group if not f.flagged]
    if no_flag:
        # フラグなしの中から最高スコア
        best = max(no_flag, key=lambda f: f.score)
        return Selection(best, reason="clean_max_score")
    else:
        # 全てフラグあり → 最高スコア (要確認のまま)
        best = max(group, key=lambda f: f.score)
        return Selection(best, reason="flagged_max_score", warn=True)
```

同点の場合: ファイルサイズ大 (高ビットレート/解像度) を優先。

---

## 4. メタデータ取得

### 4.1 抽出フィールド (mutagen)

| フィールド | Vorbis Comment タグ | 必須/任意 | 不足時 fallback |
|---|---|---|---|
| artist | `artist` | 必須 | albumartist → "Unknown Artist" |
| albumartist | `albumartist` | 任意 | artist で代替 |
| title | `title` | 必須 | ファイル名から推定 |
| album | `album` | 任意 | "Unknown Album" |
| date | `date` | 任意 | "" |
| genre | `genre` | 任意 | "Unknown" |
| discnumber | `discnumber` | 任意 | "1" |
| tracknumber | `tracknumber` | 任意 | "00" |
| circle | `circle` | 任意 | "" (同人系) |
| brand | `brand` | 任意 | "" (ゲーム系) |
| series | `series` | 任意 | "" |

### 4.2 ファイル名フォールバックパース

`{tracknum}. {title}` または `{tracknum} {title}` を正規表現で抽出。
パース不可なら `os.path.splitext(basename)[0]` を title に代入。

---

## 5. opus エンコード

### 5.1 コマンド

```bash
ffmpeg -i {input.flac} \
  -c:a libopus -b:a 256k -vbr on -compression_level 10 -application audio \
  -map_metadata 0 \
  {output.opus}
```

- bitrate 256k VBR: 透明圧縮目安 (48kHz サンプル, ステレオ)
- compression_level 10: 最高品質 (CPU 時間長いがバッチ運用なので許容)
- application audio: 音楽特化モード

### 5.2 アートワーク移植

ffmpeg の `-map_metadata` では PICTURE が opus に移らないため、mutagen で別途処理:

```python
# 入力 FLAC から抽出
pictures = _extract_flac_pictures(input_flac)
# opus 出力に書き込み (mutagen.oggopus.OggOpus)
opus = OggOpus(output_opus)
for pic in pictures:
    opus["metadata_block_picture"] = [base64_encode_picture(pic)]
opus.save()
```

opus の PICTURE は base64 化された PICTURE ブロックを `metadata_block_picture` Vorbis Comment に入れる仕様。

### 5.3 DSRE 識別タグ

opus 出力にも `dsre_version` / `dsre_processed_utc` を埋め込む。`_embed_output_metadata_opus(path)` ヘルパーを追加。

---

## 6. foobar 互換フォルダ階層

### 6.1 デフォルトテンプレート

```
{OUTPUT_DIR}\Audio\{category}\{artist_or_circle}\{album}\{disc_track}.{title}.opus
```

- `category`: genre タグから category mapping table 経由
- `artist_or_circle`: circle タグがあれば circle、なければ albumartist (なければ artist)
- `disc_track`: disc=1 のみなら `{tracknum:02d}`、複数 disc なら `{disc}-{tracknum:02d}`

### 6.2 category mapping table

```python
GENRE_CATEGORY = {
    "j-pop": "J-Pop",
    "anime": "Anison",
    "anison": "Anison",
    "game": "VGM",
    "vgm": "VGM",
    "classical": "Classical",
    "doujin": "Doujin",
    "touhou": "Doujin",
    "vocaloid": "Vocaloid",
    # ... fallback
    "_default": "Other",
}
```

genre が table にない場合は "Other" へ。table は外部 JSON (`genre_category.json`) に切り出し、ユーザー編集可能とする。

### 6.3 Windows ファイル名サニタイズ

禁止文字 `< > : " / \ | ? *` を全角に置換:
- `<` → `＜`, `>` → `＞`, `:` → `：`, `"` → `”`, `/` → `／`, `\` → `＼`, `|` → `｜`, `?` → `？`, `*` → `＊`

末尾の `.` や空白も削除 (Windows 仕様)。

### 6.4 衝突回避

同パスが既に存在する場合:
- ハッシュ比較し同一なら skip
- 異なる内容なら suffix `_2.opus`, `_3.opus`, ...

---

## 7. 処理順序

グループ単位での走査順:
```
ORDER BY albumartist, album, discnumber, tracknumber
```

同アルバムを連続処理することで:
- I/O ローカリティ向上
- 進捗表示の自然さ (album 単位で完了)
- アートワーク Picture の重複読込最小化 (将来キャッシュ可能)

---

## 8. UI 変更 (MainWindow)

「常に全自動」のため、UI は最小限。

### 8.1 既存 UI 維持

- ラベル (進捗テキスト)
- pb_file (現ファイル進捗)
- pb_all (全体進捗)
- btn_start (開始)
- 負荷スライダー

### 8.2 進捗テキスト形式

```
解析中 (12/45)
グルーピング: 28 group, 17 重複
処理中 [3/28] album_name / track_03.flac
opus 変換中 ...
仕分け中 → Audio/J-Pop/...
完了 22/28 (4 skip, 2 ⚠要確認)
```

### 8.3 要確認ダイアログ (フラグ時のみ)

全候補にフラグ ON のグループに当たった時、モーダルで:
- グループ内全ファイルのスコア・フラグ一覧表
- 「最高スコアで進める」/「このグループをスキップ」/「全 ⚠ を一律進める」(以降同セッション無確認)

「全 ⚠ を一律進める」は session 終了で reset。

### 8.4 ログ出力

詳細ログを `OUTPUT_DIR/_workflow_log.txt` に追記:
```
[2026-05-21T18:00:00] group=Author-Title-Album files=3 selected=track01_v3.flac score=78 reason=clean_max_score
[2026-05-21T18:00:42] -> Audio/J-Pop/Artist/Album/01.Title.opus
[2026-05-21T18:00:42] trash: track01_v1.flac (score=45 flagged: brick_wall)
[2026-05-21T18:00:42] trash: track01_v2.flac (score=62 ok)
```

---

## 9. 新規クラス・モジュール

| クラス | 責務 | 既存依存 |
|---|---|---|
| `WorkflowScanner` | INPUT_DIR 走査・メタデータ抽出 | mutagen |
| `TrackGrouper` | 同曲グルーピング (3 段階判定) | rapidfuzz |
| `QualityScorer` | スコア式 + フラグ判定 | 既存 MetricsComputer |
| `BestSelector` | グループ内最良選択 | QualityScorer |
| `OpusEncoder` | DSRE 後 FLAC → opus 変換 + アートワーク移植 | ffmpeg, mutagen |
| `FoobarSorter` | メタデータ → foobar 階層パス生成 | (純関数) |
| `WorkflowOrchestrator` | 全体 DAG 統括 | 全部 |

`WorkflowOrchestrator` が `Worker.run` の置換対象。既存 `_process_one` は最良ファイル単体処理関数として温存し、Orchestrator から呼ぶ。

---

## 10. 依存追加

```
rapidfuzz>=3.6.0      # ファイル名 fuzzy match (グルーピング段階 3)
```

opus 変換は同梱 ffmpeg を流用。新規バイナリ依存なし。

DSRE.spec の hidden imports / datas に `rapidfuzz` 追加。

---

## 11. 変更ファイル

| ファイル | 変更種別 | 概要 |
|---|---|---|
| `DSRE.py` | 大幅追加 | 7 新クラス、Worker 置換、Orchestrator 統括 |
| `requirements.txt` | 修正 | `rapidfuzz` 追加 |
| `DSRE.spec` | 修正 | `rapidfuzz` hidden imports / collect_all 追加 |
| `genre_category.json` | 新規 | ジャンル → category マッピング (ユーザー編集可) |
| `version_tags.json` | 新規 | サフィックス検出パターン (将来拡張用) |

---

## 12. テスト計画

| 検証項目 | 方法 | 合格基準 |
|---|---|---|
| グルーピング 完全一致 | mock metadata 3 同曲 | 1 group |
| グルーピング 別バージョン | "[Live]" 含む 2 件 | 2 group |
| グルーピング 部分一致 | duration ±1 秒の同曲 | 1 group |
| スコア式 高品質 | DR=15 LUFS=-14 → score 計算 | score > 80 |
| スコア式 過剰圧縮 | DR=4 plr=3 → score 計算 | score < 30 |
| フラグ brick wall | flatness=0.03 | flagged=True |
| 最良選択 全 clean | clean 3 件 | max score 選択 |
| 最良選択 全 flagged | flagged 3 件 | max + warn=True |
| opus 変換 | 96kHz FLAC → opus | 256k VBR, 復号可 |
| アートワーク移植 | JXL 入り FLAC → opus | PICTURE 保持 |
| foobar パス生成 | full metadata | template 通り |
| foobar パス サニタイズ | `?` 入り title | 全角化 |
| 衝突回避 | 同パス既存 | `_2.opus` 等 |
| selftest | `--selftest` | verdict=EQUIV |

---

## 13. ロールバック・段階的有効化

環境変数 `DSRE_WORKFLOW=0` で旧パイプライン (sub-project A 状態) に強制 fallback。デフォルトは新パイプライン。

問題発生時は env var 1 行で旧動作復帰、コード変更不要。

---

## 14. リスク と 緩和策

| リスク | 緩和 |
|---|---|
| 重要バージョンを誤判定で破棄 | send2trash 使用、復元可能 |
| メタデータ不一致で同曲統合失敗 | 段階 2 / 3 のフォールバック判定 |
| opus 変換でアートワーク欠落 | mutagen.oggopus で明示移植 |
| foobar 互換性問題 | category mapping を外部 JSON 化 |
| 主観劣化を検出できない | 要確認ダイアログ + ログで履歴保持 |
| 処理時間激増 (解析 + opus 変換) | グループ単位並列化 (将来) |

---

## 15. Out of Scope (将来 sub-project)

- **Sub-project C**: メタデータ自動取得 (VGMdb / MusicBrainz)。本設計は「メタデータが既にある」前提
- **acoustic fingerprint (chromaprint)**: 段階 2/3 で不十分な場合の最終手段
- **並列化**: グループ単位で thread pool
- **ユーザー学習**: 「以前 user が選んだバージョン」を覚えて優先

---

## 16. 設定可能パラメータ (env var)

| env var | デフォルト | 説明 |
|---|---|---|
| `DSRE_WORKFLOW` | `1` | 0 で旧パイプライン |
| `DSRE_OPUS_BITRATE` | `256k` | opus 出力ビットレート |
| `DSRE_SCORE_LUFS_TARGET` | `-14` | LUFS 目標値 |
| `DSRE_AUTO_CONFIRM_FLAGGED` | `0` | 1 で全 ⚠ 無確認実行 |
| `DSRE_KEEP_INTERMEDIATE_FLAC` | `0` | 1 で中間 96kHz FLAC を保持 |
