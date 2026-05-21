# DSRE Sub-project B+D: ワークフロー自動化 + 客観品質ゲート 設計書 (v5)

**Date:** 2026-05-21 (v5 抽象化・例の literal 除去・採点シンプル化)
**Core principle:** 同曲識別は音響指紋で行い、メタデータは指紋クラスタ内で自動伝播する。採点は clip=0 正規化 PCM の直接解析。最小手動介入で raw 投入 → 整列出力。

---

## 0. 出力形式

| 種類 | 形式 | 役割 |
|---|---|---|
| **最終出力** | FLAC 96kHz PCM_24 | DSRE 処理結果 (既存通り) |

採点用 tmp ファイルは生成しない (§4 参照、opus 採点を廃止)。

---

## 1. ユーザーワークフロー (前提)

```
raw FLAC 群 (同曲の異なる版が混在)
   ↓ 一部だけ mp3tag で手動タグ取得 (残りはタグ未整備で可)
   ↓ DSRE 用フォルダに全部投入                ← B+D が以下を全自動化
   ↓ DSRE 処理
   ↓ アートワーク埋め込み (Sub-A 済)
   ↓ foobar 互換階層への仕分け
   ↓ foobar 経由で opus 変換 (DSRE 外)
```

ユーザー思想: **手動確認と所要時間を極限まで減らす**。タグ未整備の版があっても、整備済の版から自動補完して同じ品質の整列が達成される。

---

## 2. 全体パイプライン

```
[STAGE 1: 処理前]
  INPUT_DIR 再帰スキャン (既処理 = dsre_version タグ持ち は skip)
    ↓
  Acoustic fingerprint 取得 (chromaprint、content-hash SQLite cache)
    ↓
  指紋クラスタリング (類似度閾値、union-find)
    ↓
  各クラスタで canonical 選定 (タグ充実度 + アートワーク有無の加重)
    ↓
  canonical → クラスタ内他 file へメタデータ + アートワーク 伝播
    (version 識別系タグと既値は不変)
    ↓
  版分岐 (version 識別系タグの差分で sub-cluster へ分割、両方残す)
    ↓
  各 sub-cluster で各 file を採点 (clip=0 正規化 → 直接解析)
    ↓
  各 sub-cluster で最良 1 つを選択 (要注意フラグ判定込み)
    ↓
  非最良は識別可能リネーム + send2trash
    ↓
  最良 file 群を foobar 階層で INPUT_DIR 内に再配置

[STAGE 2: DSRE 処理]
  最良 file 群を順次 DSRE 処理 (zansei_impl、Sub-A 含む既存パス)

[STAGE 3: 処理後]
  出力 FLAC を foobar 階層で OUTPUT_DIR 内に配置
  → INPUT_DIR と OUTPUT_DIR は同型レイアウト
```

---

## 3. 同曲識別

### 3.1 Acoustic Fingerprint

chromaprint (`fpcalc.exe`、ffmpeg と並列で DSRE バンドル) を subprocess で呼び、各 file の指紋を取得する。指紋は楽曲の音響的内容に基づく識別子で、メタデータの有無に依存しない。

### 3.2 指紋キャッシュ

既存 `dsre_log.db` に新規テーブル `fingerprints (content_hash, duration_sec, fingerprint, last_path, computed_at)` を追加。content_hash = `md5(file_size + 先頭 64KB + 末尾 64KB)`。再走時の指紋再計算スキップ。ファイル移動・リネーム後も hash 一致で cache 適用。

### 3.3 クラスタリング

全 file pairwise で指紋 Hamming 一致率を計算し、閾値以上のペアを union-find で連結。閾値は env で調整可能 (デフォルト 0.85、保守側)。

### 3.4 メタデータ伝播

各クラスタ内:

1. **canonical 選定** — タグ充実度を以下の加重で評価し最大値を持つ file:
   - 主要タグ (artist, album, title, date, discnumber, tracknumber) 各 2 点
   - 拡張タグ (genre, circle, brand, project, ...) 各 1 点
   - アートワーク存在 3 点
   - 同点は file サイズ大、それも同点なら path 辞書順 (決定的)
2. **伝播対象** — 主要タグ + 拡張タグ + アートワーク (PICTURE ブロック)
3. **絶対不変 (version 識別系)** — `version_info`, `cover_type`, `live_type`, `vocal_type`, `remaster_info`, `arrange_type`, `m_number`, `featuring`, `produced`
4. **既値の尊重** — 伝播先 file が既に該当タグを持つ場合は上書きしない (ユーザー意図保護)
5. **書き込み** — mutagen で in-place 更新、各 file の dsre_version タグはまだ立てない (STAGE 2 完了時に立つ)

### 3.5 版分岐

伝播後、クラスタ内で version 識別系タグの組が異なる file を別 sub-cluster に分離する。同 sub-cluster 内は「version 含めて完全同一の真の重複」→ 採点で 1 つに絞る。複数 sub-cluster は別バージョンとして全て採点・処理対象。

### 3.6 失敗時の縮退

指紋取得失敗 (corrupt audio、binary 不在) → 該当 file は単独クラスタとして処理を継続。フラグ機能の上位が無効化されるだけで、DSRE 処理自体は失われない。

---

## 4. 客観品質スコアリング

### 4.1 採点の本質

ユーザー要件: 「clip=0 の最終状態で比較したい」。これは「音量に依存した clip 数の偏差を除去して各 file の本質的な音響特性を比べる」ことに等しい。clip=0 を達成すれば opus エンコードの有無で結果はほぼ変わらない。よって **採点用 opus 変換は行わない**。

### 4.2 採点手順

```
1. file 読み込み
2. DSRE 既存 volume normalization を流用 (save_flac24_out から抽出)
   → true peak -0.3 dBFS で clip=0 を構造保証
3. 正規化済 PCM を MetricsComputer.compute() で 15 メトリクス取得
4. スコア式を適用
5. フラグ判定
```

### 4.3 リファクタ要件

既存 `save_flac24_out` 内の音量正規化処理を独立関数 `_dsre_normalize_volume(audio, sr) -> audio_norm` として抽出する。既存呼出側 (write 用) と新規採点側が同一実装を共有することで、採点と本処理の乖離リスクを構造的に排除する。

### 4.4 スコア式 (初期値、env で調整可能)

```python
score = 0.0
score += clamp(dr, 0, 20) * w_dr               # max 60
score += clamp(20 - abs(lufs - lufs_target), 0, 20)  # max 20
score += clamp(hf_ratio_12k, 0, 0.05) * 200    # max 10
score += clamp(flatness, 0, 0.5) * 20          # max 10
score -= max(0, plr_floor - plr) * 2           # 最大 -12
score = clamp(score, 0, 100)
```

clip_count / peak_db は normalize で揃うため減点対象外。係数は実音源で観測しつつ env で調整。

### 4.5 要注意フラグ (D: 数値良好でも実聴劣化)

| 条件 | 検出意図 |
|---|---|
| `hf_ratio_16k` 極小 かつ `rolloff_hz < 17000` | 高域カット (lossy origin) |
| `flatness < 0.05` | brick wall master |
| `dr` 危険水準 | ハイパー圧縮 |
| 高 `harmonic_1k_proxy` かつ 高 `dr` | 高 DR だが歪み多 |
| 不自然な高域偏重 (centroid 高 + hf_8k 大) | 人工アップサンプル疑い |

各閾値は env で調整可能。フラグはログのみ。

### 4.6 最良選択

フラグ無し集合の最高スコアを優先。フラグ無し集合が空ならフラグ付き集合から最高スコア (警告ログ)。同点はファイルサイズ大優先。

---

## 5. ユーザー介入最小化

デフォルトはフラグ付きクラスタも警告ログのみで自動進行。`DSRE_INTERACTIVE_CONFIRM=1` で初めて UI 確認 (opt-in)。事後レビューは `OUTPUT_DIR/_workflow_log.txt`、復元はゴミ箱から。

---

## 6. foobar 互換フォルダ階層

### 6.1 階層構築ルール

参考にしたユーザー foobar config の構造を、明示的ルールに再表現する。

**1. カテゴリレベル (タグ存在で挿入、欠落で省略):**
```
genre / age / circle / category / source / grouping
```

**2. 作品識別レベル (親子重複排除を伴うタグ存在挿入):**
```
franchises / products / series / brand / subtitle / elements
```
親子重複排除ルール:
- `franchises = project` なら franchises 省略
- `products = project` または `products = brand` なら products 省略
- `series = project` なら series 省略
- `subtitle = brand` または `subtitle = franchises` なら subtitle 省略
- `elements = brand` または `elements = franchises` または `elements = subtitle` なら elements 省略

これは「ある分類軸で値が等しい場合に下位レベルを冗長に作らない」ためのルール。

**3. プロジェクト・グルーピングレベル:**
```
project / collaboration / group / unit / album_type
```

**4. 時系列レベル (date タグから派生、date 存在時のみ):**
```
year(date) / month(date)
```

**5. アルバム + ディスク:**
```
{album}[" (" + date + ")"]      ← date 存在で suffix 追加
Disc {disc:02d}                ← 単 disc アルバムでは省略
```

**6. ファイル名:**
```
{disc:02d}.{track:03d}.{title}{artist 部}{修飾 [..] 群}.flac
```

artist 部:
- `featuring` 存在: ` - {artist} [feat. {featuring}]`
- 不在で artist 存在: ` - {artist}`
- artist も不在: 省略

修飾 `[..]` 群 (各タグが存在する分だけ列挙、不在で省略):
```
[Prod. {produced}]
[{arrange_type}]
[{version_info}]
[{remaster_info}]
[{cover_type}]
[{live_type}]
[{vocal_type}]
[{m_number}]
```

### 6.2 OUTPUT_DIR 配下のパス組立

OUTPUT_DIR (`C:\Audio\DSRE\Output\`) には既に `Audio/` が含まれるため、foobar 原テンプレ先頭の `Audio/[{device}]/` プレフィックスは DSRE 側では使わない。§6.1 のレベル 1 から直接展開する。

### 6.3 複数値タグの delimiter 置換

`artist` の `/` → `_ `、`featuring` の `/` → `* `、修飾 [..] 内の `;` → ` /`。foobar 原テンプレ準拠。

### 6.4 衝突回避

同パス既存:
- 内容ハッシュ一致 → skip + ログ
- 相違 → 数字 suffix `_2.flac`, `_3.flac`, ...

### 6.5 Windows 制約

- 禁止文字 `< > : " / \ | ? *` を全角に置換
- 末尾 `.` / 空白を削除
- 250 chars 超過時 `\\?\` プレフィックス + 警告ログ

### 6.6 INPUT/OUTPUT 同型保証

STAGE 1 (処理前) と STAGE 3 (処理後) は §6.1 の同一テンプレを使う。結果としてユーザーが INPUT_DIR を覗いても OUTPUT_DIR を覗いても 1:1 対応で同じ階層が見える。

---

## 7. 破棄処理

非最良 file は send2trash の前に識別可能リネーム:
```
[DSRE-inferior] {元のファイル名}.flac
```

`[DSRE-inferior]` プレフィックスにより、ゴミ箱で名前順に並べたとき先頭付近に固まる + 「DSRE が劣品質と判定して捨てた」と即識別可能。破棄理由・スコア・フラグは `OUTPUT_DIR/_workflow_log.txt` に追記。

---

## 8. 冪等性と再開

冪等性: 各 file の `dsre_version` Vorbis Comment タグを確認 → 存在すれば skip。何度開始しても破壊しない。

再開: journal なし。状態は filesystem + tag が自己記述する。任意の STAGE での中断後、再走で続きから進む。

---

## 9. 処理順序

ORDER BY artist, album, discnumber, tracknumber。I/O ローカリティと進捗の自然さ。

---

## 10. UI

既存維持。進捗テキストは STAGE と件数を構造化して表示する。要確認ダイアログはデフォルト OFF (§5)。

---

## 11. 新規モジュール

| クラス/関数 | 責務 |
|---|---|
| `FingerprintEngine` | fpcalc subprocess + SQLite cache |
| `ClusterBuilder` | pairwise 類似度 + union-find |
| `MetadataExtractor` | mutagen による多タグ抽出 |
| `MetadataPropagator` | canonical 選定 + 伝播 + 既値尊重 + 版分岐 |
| `QualityProbe` | 正規化 + 解析 (opus なし、§4.2) |
| `BestSelector` | sub-cluster 内最良選択 |
| `DiscardHandler` | 識別可能リネーム + send2trash |
| `FoobarPathBuilder` | §6.1 のルールに基づくパス構築 + サニタイズ + 長パス対応 |
| `WorkflowOrchestrator` | STAGE 1-3 統括 |

リファクタ: `save_flac24_out` 内 volume normalization 抽出 → `_dsre_normalize_volume(audio, sr)` (純関数化、既存呼出側と新規 `QualityProbe` 両方が呼ぶ)。

---

## 12. 依存

### バイナリ
- `fpcalc.exe` (chromaprint 公式) を DSRE バンドルに追加 (~500KB)、`_internal/ffmpeg/` 配下

### Python
- 追加なし。fpcalc は subprocess、指紋比較は自前実装 (decode + Hamming + best-alignment)

### DSRE.spec
- `fpcalc.exe` の datas 配置追加

---

## 13. 変更ファイル

| ファイル | 変更種別 | 概要 |
|---|---|---|
| `DSRE.py` | 大幅追加 + 軽微リファクタ | 9 新モジュール、Worker 改修、Orchestrator 統括、`save_flac24_out` の volume norm 抽出、SQLite 新テーブル |
| `DSRE.spec` | 修正 | `fpcalc.exe` バンドル |
| `_internal/ffmpeg/fpcalc.exe` | 新規配置 | chromaprint 公式 binary |

---

## 14. テスト計画

| 検証項目 | 合格基準 |
|---|---|
| 指紋計算 | duration + fp 取得 |
| 指紋キャッシュ | 2 回目は cache hit |
| クラスタ similarity 高 | 同 audio で類似率 > 0.95 |
| クラスタ separation | 別曲で類似率 < 0.5 |
| Union-find 連鎖 | 推移的連結 |
| canonical 選定 | 加重で最多タグ持ち選出 |
| メタデータ伝播 | 未タグ file に canonical のタグ + アートワーク |
| version 系タグ保護 | 触らない |
| 既値尊重 | 上書きしない |
| 版分岐 | version 系タグ差分で sub-cluster 分離 |
| 採点 normalize | clip=0 達成 |
| 採点 高品質 mock | score > 80 |
| フラグ各種 | 該当条件で立つ |
| 最良選択 全 clean | max score |
| 最良選択 全 flagged | max + warn |
| 破棄リネーム | `[DSRE-inferior]` プレフィックス |
| ゴミ箱投入 | send2trash 成功 |
| FoobarPath 各種 (full / partial / sanitize / 長パス / delimiter) | テンプレ規則通り |
| 単 disc 省略 | Disc 階層なし |
| 衝突回避 | 数字 suffix |
| 冪等性 | 既処理 file skip |
| 再開 (STAGE 2 中断) | 再走で続き処理 |
| 統合 (multi-version mock) | INPUT/OUTPUT 同型 + 破棄 file 識別可能 |
| selftest | verdict=EQUIV |

---

## 15. 設定可能パラメータ (env)

| env var | デフォルト | 説明 |
|---|---|---|
| `DSRE_WORKFLOW` | `1` | 0 で旧パイプライン (Sub-A 状態) |
| `DSRE_PRESORT_INPUT` | `1` | 0 で STAGE 1 仕分け skip |
| `DSRE_CLUSTER_SIMILARITY` | `0.85` | 指紋類似度閾値 |
| `DSRE_HARMONIZE_METADATA` | `1` | 0 で伝播無効 |
| `DSRE_INTERACTIVE_CONFIRM` | `0` | 1 でフラグ群に UI ダイアログ |
| `DSRE_DISC_DIR_ALWAYS` | `0` | 1 で単 disc でも `Disc 01/` |
| `DSRE_SCORE_LUFS_TARGET` | `-14` | LUFS 目標値 |
| `DSRE_SCORE_WEIGHT_DR` | `3.0` | DR 係数 |

---

## 16. リスク と 緩和

| リスク | 緩和 |
|---|---|
| 指紋誤クラスタ (false positive) | 閾値高め (0.85) + 内容ハッシュ完全一致時のみ「真の重複」と扱う |
| 指紋取りこぼし (false negative) | 単独クラスタとして処理続行 (損失なし) |
| canonical 誤判定 | 既値尊重ルール + 伝播ログ |
| version 系タグ消失 | 不変リストで構造保護 |
| アートワーク喪失 | mutagen clear+add 順序、Sub-A の知見流用 |
| fpcalc.exe 不在 | 警告 + 指紋機能無効化 (単独クラスタ扱いで処理続行) |
| Windows 長パス | `\\?\` + 警告 |
| 採点と本処理の正規化乖離 | 同一関数の共有 (リファクタで構造保証) |
| 「数値良好で実聴劣化」 | 5 フラグ + 任意 opt-in 確認 |

---

## 17. Out of Scope

- **Sub-project C**: メタデータ自動取得 (AcoustID/MusicBrainz/VGMDB/iTunes/Apple Music)。本設計は「一部 file がタグ済」を前提とするが、C を実装すれば「全 file 未タグ」状態でも完結可能になる。
- **foobar 経由 opus 変換**: ユーザー従来通り。

---

## 18. 設計核心 (1 文)

> **同曲識別を音響指紋に基盤を移し、メタデータをクラスタ内伝播で自動補完し、採点を clip=0 正規化 PCM の直接解析にすることで、タグ整備が不完全な raw 投入から整列出力までを最小手動介入で達成する。**
