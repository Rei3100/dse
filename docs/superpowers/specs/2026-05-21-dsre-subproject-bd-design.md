# DSRE Sub-project B+D: ワークフロー自動化 + 客観品質ゲート 設計書 (v4)

**Date:** 2026-05-21 (v4 構造再設計: acoustic fingerprint を中核に)
**Core Insight:** メタデータが揃わない現実 (ユーザーは複数版のうち 1 版だけ手動タグ付け) を構造的に解決するため、**chromaprint (acoustic fingerprint) を曲識別の primary key とし、メタデータは fingerprint cluster 内で自動伝播させる**。

---

## 0. 出力形式

| 種類 | 形式 | 役割 |
|---|---|---|
| **最終出力** | **FLAC 96kHz PCM_24** | DSRE 処理結果。既存通り。これが残る。 |
| **opus 採点** | **テンポラリ** | 同条件比較のためだけに生成、解析後即削除 |

opus は DSRE の最終出力ではない。foobar 経由の opus 変換は **ユーザーが従来通り別途**。

---

## 1. ユーザーの実ワークフロー (前提・原指示準拠)

```
音源 (raw FLAC、同曲複数版あり)
   ↓ mp3tag で 1 版のみメタデータ取得 (10 曲 × 3 版 = 30 file のうち 10 file だけタグ済)
   ↓ DSRE 用フォルダに 30 file 全部投入 ← B+D が以下を全自動化
   ↓ DSRE 処理
   ↓ アートワーク埋め込み (Sub-A 済)
   ↓ foobar 互換階層仕分け
   ↓ foobar で opus 変換 (DSRE 外)
```

ユーザー思想 (原指示): **手動確認・かかる時間を極限まで減らす**。

---

## 2. 全体パイプライン

```
[STAGE 1: 処理前段階]
  INPUT_DIR 再帰スキャン (.flac)
    ↓ dsre_version タグ持ち = 既処理、skip (冪等)
  Acoustic fingerprint 計算 (chromaprint、結果は SQLite キャッシュ)
    ↓
  Fingerprint クラスタリング (類似度 > 閾値で union-find)
    ↓
  各 cluster で canonical metadata 源を特定 (タグ充実度最大)
    ↓
  Cluster 内で canonical → 他 file へメタデータ + アートワーク 伝播
    (version 系タグ群は伝播対象外、各 file 独自を尊重)
    ↓
  バージョン分離 (cluster 内で version 系タグ差分のある file は別 sub-cluster に分割、両方残す)
    ↓
  各 sub-cluster で各 file を採点 (opus 182k 並列、DSRE 同等 normalize)
    ↓
  各 sub-cluster で最良 1 つを選択 (要注意フラグ判定込み)
    ↓
  非最良は識別可能リネーム + send2trash
    ↓
  最良 file 群を foobar 互換階層で INPUT_DIR 内に再配置

[STAGE 2: DSRE 処理]
  最良 file 群を順次 DSRE 処理 (zansei_impl、既存)
    ↓ アートワーク埋め込み (Sub-A、既存)
  96kHz FLAC PCM_24 出力

[STAGE 3: 処理後段階]
  出力 FLAC を foobar 互換階層で OUTPUT_DIR 内に配置
  → INPUT_DIR と OUTPUT_DIR が 1:1 同型レイアウト
```

---

## 3. 同曲識別 (B+D の構造的核)

### 3.1 Acoustic Fingerprint (chromaprint)

`fpcalc.exe` (chromaprint 同梱バイナリ、ffmpeg と並列に DSRE バンドル) を subprocess で呼び、各 file の fingerprint を取得:

```bash
fpcalc -json {file.flac}
# → {"duration": 245.3, "fingerprint": "AQAAB0kkRYqYREqOpEn..."}
```

fingerprint は楽曲の音響的指紋 (MusicBrainz Picard / foobar2000 が使うのと同じ規格)。**メタデータ無関係に「同じ録音」を識別**。

### 3.2 Fingerprint キャッシュ (SQLite)

既存 `C:\FreeSoft\DSRE\dsre_log.db` に新規テーブル:

```sql
CREATE TABLE IF NOT EXISTS fingerprints (
    content_hash TEXT PRIMARY KEY,
    duration_sec REAL,
    fingerprint TEXT,
    last_path TEXT,
    computed_at TEXT
);
```

- `content_hash`: `md5(file_size_bytes + first_64KB + last_64KB)` で content-addressing
- 再実行時は cache hit → fingerprint 再計算スキップ
- ファイル移動・リネームは content_hash が同じなので影響なし

### 3.3 クラスタリング (Union-Find)

全 file pairwise で fingerprint 比較し、類似度 > 閾値ペアを union-find で連結:

```python
def similarity(fp_a: list[int], fp_b: list[int]) -> float:
    """chromaprint fingerprint の Hamming 一致率 (0.0-1.0)"""
    # 短い方を長い方に sliding window で重ね、最大一致率を返す
    # bit-level XOR + popcount
```

閾値デフォルト: **0.85** (env `DSRE_CLUSTER_SIMILARITY` で調整可能)。

- 通常: 同曲の異なる版 (同マスター・別エンコード) は 0.90+ で一致
- リマスター違い: 0.85-0.95
- 完全別曲: 0.30 以下
- 0.85 は false positive と false negative のバランス値

### 3.4 メタデータ伝播 (canonical → cluster 内全 file)

各 cluster 内で以下:

1. **canonical 選定**: 「タグ充実度」最大の file を canonical とする
   - 充実度スコア = 主要タグ (artist, album, title, discnumber, tracknumber, date, genre) の存在数 + アートワーク有無 (+1) + ジャンル系タグ存在 (+0.5 each)
   - 同点なら mtime 新しい方
2. **伝播対象タグ** (cluster 内全 file の対象タグが空 or 全 file 同一値 = 伝播):
   - artist, album, title, discnumber, tracknumber, date, genre, age, circle, category, source, grouping, franchises, products, series, brand, subtitle, elements, project, collaboration, group, unit, album_type
   - アートワーク (PICTURE ブロック)
3. **絶対に伝播しない (version 識別系)**:
   - `version_info`, `cover_type`, `live_type`, `vocal_type`, `remaster_info`, `arrange_type`, `m_number`
   - `featuring`, `produced` (版ごとに変動する可能性)
4. **既に値を持つタグは尊重**: cluster 内 file が canonical と異なる値を既に持つ → 上書きしない (ユーザー意図保護)

伝播後、cluster 内全 file は: 共通タグ統一 + アートワーク統一 + version 系タグだけ各自独自。

### 3.5 バージョン分離 (sub-cluster)

メタデータ伝播後、cluster 内で **version 系タグの値が異なる file** は別 sub-cluster に分離:

```python
def sub_cluster_key(file) -> tuple:
    return tuple(file.tags.get(k, "") for k in [
        "version_info", "cover_type", "live_type", "vocal_type",
        "remaster_info", "arrange_type", "m_number",
    ])
```

同 sub-cluster 内の file は「真の重複」(version 系含め完全同一) → 1 つに絞る。
sub-cluster が 2 個以上 → 別バージョンとして両方処理 (それぞれが独立に最良選択+処理)。

### 3.6 5-key 整合検査 (sanity check)

メタデータ伝播後、sub-cluster 内全 file の (artist, album, disc, track, title) が一致しているか確認:
- 一致 → 通常進行
- 不一致 → ログ警告 (伝播ロジックの bug 兆候、ユーザー側の手動タグ不整合の検出)

### 3.7 Fingerprint 失敗・無効ファイル

`fpcalc.exe` 失敗 (corrupt audio, unsupported format):
- 該当 file は orphan singleton として扱う (cluster サイズ 1)
- 警告ログ
- 通常処理続行 (DSRE 処理は試みる)

---

## 4. 客観品質スコアリング

### 4.1 採点用 opus エンコード

音量正規化は **DSRE 既存の `save_flac24_out` 内 volume normalization をリファクタ抽出して流用** (true peak 8x oversampling → -0.3 dBFS target):

```python
audio_norm = _dsre_normalize_volume(audio, sr)  # 既存ロジック抽出
# tmp FLAC 経由で ffmpeg libopus 182k VBR
subprocess.run([ffmpeg, "-i", tmp_flac, "-c:a", "libopus",
                "-b:a", "182k", "-vbr", "on", "-application", "audio",
                tmp_opus, "-y"], check=True)
opus_audio, opus_sr = sf.read(tmp_opus)
metrics = MetricsComputer.compute(opus_audio, opus_sr)
score = calculate_score(metrics)
# try/finally で tmp 削除
```

DSRE と同じ normalize → 「DSRE 処理後 opus 変換した最終形」を限りなく忠実にシミュレート。フェア比較の本質。

### 4.2 並列化

`ThreadPoolExecutor(max_workers=os.cpu_count())`。tmp は per-future の try/finally で確実削除。

### 4.3 スコア式 (初期値、env で調整可)

```python
score = 0.0
score += clamp(dr, 0, 20) * 3.0               # max 60
score += clamp(20 - abs(lufs + 14), 0, 20)     # max 20
score += clamp(hf_ratio_12k, 0, 0.05) * 200    # max 10
score += clamp(flatness, 0, 0.5) * 20          # max 10
score -= max(0, 6 - plr) * 2                   # max -12
score = clamp(score, 0, 100)
```

`clip_count` / `peak_db` は normalize で揃うため減点対象外。

### 4.4 要注意フラグ (D: 数値良好でも実聴劣化)

| 条件 | 検出意図 |
|---|---|
| `hf_ratio_16k < 0.0005` かつ `rolloff_hz < 17000` | 高域カット (lossy origin 疑い) |
| `flatness < 0.05` | brick wall master |
| `dr < 6` | ハイパー圧縮原音 |
| `harmonic_1k_proxy > 0.5` かつ `dr > 12` | アーティファクト混入 |
| `hf_ratio_8k > 0.3` かつ `centroid_hz > 6000` | 人工アップサンプル疑い |

フラグはログのみ。デフォルトは確認なし自動進行。

### 4.5 最良選択

```python
def select_best(sub_cluster):
    pool = [f for f in sub_cluster if not f.flagged] or sub_cluster
    return max(pool, key=lambda f: (f.score, f.file_size))
```

---

## 5. ユーザー介入の最小化

- **デフォルト**: 全 cluster 自動進行 (フラグ付きも警告ログだけで処理続行)
- **`DSRE_INTERACTIVE_CONFIRM=1`**: 全フラグ群に UI 確認 (opt-in)
- 事後レビュー: `OUTPUT_DIR/_workflow_log.txt` で全判断を追跡可能、ゴミ箱から復元可

---

## 6. foobar 互換フォルダ階層

### 6.1 DSRE 適用テンプレ (OUTPUT_DIR = `C:\Audio\DSRE\Output\` 配下、`Audio/` プレフィックスは省略)

```
{OUTPUT_DIR}/
  [{genre}/] [{age}/] [{circle}/] [{category}/] [{source}/] [{grouping}/]
  [{franchises}/] [{products}/] [{series}/] [{brand}/]
  [{subtitle}/] [{elements}/] [{project}/] [{collaboration}/]
  [{group}/] [{unit}/] [{album_type}/]
  [{year}/{month}/]
  {album}[' ('date')'] /
  [Disc {disc:02d}/]                      ← 単 disc アルバムでは省略
  {disc:02d}.{track:03d}.{title}
    [- {artist}[' [feat. 'featuring']']]
    [' [Prod. 'produced']']
    [' ['arrange_type']']
    [' ['version_info']']
    [' ['remaster_info']']
    [' ['cover_type']']
    [' ['live_type']']
    [' ['vocal_type']']
    [' ['m_number']']
  .flac
```

各 `[...]` はタグ存在で挿入、欠落で省略。foobar 原テンプレートの `$if(%tag%, ..., )` 構造に準拠。

### 6.2 複数値タグの delimiter 置換

foobar 原テンプレに従う:
- artist の `/` → `_ `
- featuring の `/` → `* `
- arrange_type, version_info, remaster_info の `;` → ` /`

### 6.3 衝突回避

同パス既存 → 内容ハッシュ一致なら skip、相違なら `_2.flac`, `_3.flac` suffix。

### 6.4 Windows パス制約

- 禁止文字 `< > : " / \ | ? *` を全角に置換
- 末尾 `.` / 空白を削除
- 250 chars 超過: `\\?\` プレフィックス使用 (MAX_PATH 回避)、警告ログ

### 6.5 単 disc 省略

album 内の全 track の discnumber が "1" のみ → `Disc 01/` 階層省略。`DSRE_DISC_DIR_ALWAYS=1` で常時作成。

---

## 7. STAGE 1: 処理前仕分け+リネーム

最良 file 群を §6.1 テンプレで INPUT_DIR 内に再配置 (rename + sort)。
ユーザーが INPUT_DIR を覗くと整理された階層が見える。STAGE 3 後と同型レイアウト。

---

## 8. 破棄処理 (識別可能リネーム → ゴミ箱)

非最良 file を send2trash 前に:
```
{元のファイル名}.__discarded_dsre_inferior__.flac
```

ユーザーがゴミ箱を覗けば「DSRE が劣品質と判定して捨てた」と即識別可能。
詳細は `OUTPUT_DIR/_workflow_log.txt` に追記。

---

## 9. STAGE 3: 処理後仕分け+リネーム

DSRE 出力 FLAC を §6.1 同テンプレで OUTPUT_DIR 内配置。
INPUT_DIR ↔ OUTPUT_DIR 1:1 同型。foobar から OUTPUT_DIR を直接読み込み既存設定で opus 変換可能。

---

## 10. 冪等性 (再実行安全)

各 file の `dsre_version` Vorbis Comment タグを確認 → 存在すれば skip。
ユーザーが何度「開始」しても破壊的変更なし。

---

## 11. 失敗時リジューム

journal なし。状態はファイル配置 + tag が自己記述:

- STAGE 1 中断: 仕分け済 file は新パスに、未処理は旧位置。再走時 `dsre_version` 無 file を再スキャン。
- STAGE 2 中断: DSRE 済 file は OUTPUT_DIR (タグ済)、未処理は INPUT_DIR の仕分け先に残置。
- STAGE 3 中断: 仕分け待ち file は OUTPUT_DIR ルート残置。再走で仕分けのみ。

冪等性 (§10) と組み合わせ、任意の中断後に完全再開。

---

## 12. 処理順序

```
ORDER BY artist, album, discnumber, tracknumber
```

I/O ローカリティ + 進捗の自然さ。fingerprint cluster は metadata 伝播後にこの並びで処理。

---

## 13. UI (MainWindow)

既存維持 + 進捗テキスト改善:
```
[STAGE 1] スキャン 30 files (0 既処理)
[STAGE 1] Fingerprinting 30/30 (cache 18 hit, 12 計算)
[STAGE 1] クラスタリング 10 cluster
[STAGE 1] メタデータ伝播 10/10 (20 file に補完)
[STAGE 1] 採点 並列 30/30
[STAGE 1] 最良選択 完了、破棄 20
[STAGE 1] 整列 INPUT_DIR
[STAGE 2] DSRE [3/10] {album}/{track}
[STAGE 3] 整列 OUTPUT_DIR
完了 10/10 (要確認 0)
```

要確認ダイアログはデフォルト OFF (§5)。

---

## 14. 新規モジュール

| クラス | 責務 | 既存依存 |
|---|---|---|
| `FingerprintEngine` | fpcalc.exe subprocess + SQLite cache | sqlite3 |
| `ClusterBuilder` | pairwise 類似度 + union-find | (純関数) |
| `MetadataExtractor` | mutagen で 25+ タグ抽出 | mutagen |
| `MetadataPropagator` | canonical 選定 + cluster 内伝播 + version 系保護 | mutagen |
| `VersionSplitter` | cluster → sub-cluster (version 系タグ差分で) | (純関数) |
| `QualityProbe` | DSRE normalize 流用 + opus encode + decode + 解析 | ffmpeg, MetricsComputer, DSRE 既存 (リファクタ抽出) |
| `BestSelector` | sub-cluster 内最良 (フラグ判定込み) | (純関数) |
| `DiscardHandler` | 識別可能リネーム + send2trash | send2trash |
| `FoobarPathBuilder` | テンプレ展開 (optional level / suffix、サニタイズ、長パス) | (純関数) |
| `WorkflowOrchestrator` | STAGE 1-3 統括、冪等性 + リジューム制御 | 全部 |

リファクタ: 既存 `save_flac24_out` 内 volume normalization 部分を `_dsre_normalize_volume(audio, sr)` として独立関数化 (既存呼出側と新規 QualityProbe 両方が呼ぶ)。

---

## 15. 依存追加

### 15.1 バイナリ
- `fpcalc.exe` (chromaprint 公式配布) を DSRE バンドルに追加 (~500KB)
- 配置: `_internal/ffmpeg/fpcalc.exe` (ffmpeg と並列)
- 検索ロジック: 既存 ffmpeg と同じ resource path 解決

### 15.2 Python (requirements.txt)
追加なし。fpcalc は subprocess で呼ぶだけ。fingerprint 比較は自前実装 (~30 行、chromaprint の base64-like 文字列を int32 list に decode → Hamming 比較)。

### 15.3 DSRE.spec
- `fpcalc.exe` のバンドルに対応するため datas にエントリ追加

---

## 16. 変更ファイル

| ファイル | 変更種別 | 概要 |
|---|---|---|
| `DSRE.py` | 大幅追加 + リファクタ | 10 新クラス、Worker 改修、Orchestrator 統括、`save_flac24_out` の normalize 抽出、SQLite 新テーブル |
| `DSRE.spec` | 修正 | `fpcalc.exe` バンドル設定追加 |
| `_ffmpeg/fpcalc.exe` | 新規配置 | chromaprint 公式 binary |
| `requirements.txt` | 変更なし | - |

---

## 17. テスト計画

| 検証項目 | 方法 | 合格基準 |
|---|---|---|
| Fingerprint 計算 | 任意 FLAC | duration + fp 取得 |
| Fingerprint キャッシュ | 2 回計算 | 2 回目は cache hit |
| クラスタ similarity | 同 audio 2 file | similarity > 0.95 |
| クラスタ separation | 別曲 2 file | similarity < 0.5 |
| Union-find | 3 file 連鎖一致 | 1 cluster |
| canonical 選定 | タグ違い 3 file | 最多タグ持ち選出 |
| メタデータ伝播 | 1 tagged + 2 untagged | 2 untagged に同タグ + artwork |
| version 系タグ保護 | live_type 異なる 2 file | 上書きされない |
| 既値尊重 | canonical と異なる genre 持ち | 上書きしない |
| sub-cluster 分割 | version_info 異 2 file | 2 sub-cluster |
| 5-key sanity 不一致警告 | mock 異常 | ログ警告出力 |
| 採点 normalize | clip 大量 file | clip=0 達成 |
| opus tmp 削除 | 通常 + 中断 | tmp 漏れなし |
| スコア値 高品質 | mock | score > 80 |
| フラグ HF cliff | rolloff=15k hf_16k=0 | flagged |
| 最良選択 全 clean | 3 件 | max score |
| 最良選択 全 flagged | 3 件 | max + フラグ維持 |
| 破棄リネーム | non-best | `__discarded_dsre_inferior__` suffix |
| FoobarPath all tags | full metadata | テンプレ完全展開 |
| FoobarPath missing | 一部欠落 | 該当 level 省略 |
| FoobarPath サニタイズ | `?` 入り | 全角化 |
| FoobarPath 長パス | >250 chars | `\\?\` プレフィックス |
| FoobarPath delimiter | artist に `/` | `_ ` 置換 |
| 単 disc 省略 | 全 disc=1 | Disc 階層なし |
| 衝突回避 | 同パス | `_2.flac` |
| 冪等性 | 既処理 file | skip |
| リジューム | STAGE 2 中断 + 再走 | 続き処理 |
| 統合 (10×3 file mock) | クラスタ → 伝播 → 処理 → 整列 | INPUT/OUTPUT 同型レイアウト + ゴミ箱に 20 識別済 file |
| selftest | `--selftest` | verdict=EQUIV |

---

## 18. 設定可能パラメータ (env var)

| env var | デフォルト | 説明 |
|---|---|---|
| `DSRE_WORKFLOW` | `1` | 0 で旧パイプライン (Sub-A 状態) |
| `DSRE_PRESORT_INPUT` | `1` | 0 で STAGE 1 の仕分け skip |
| `DSRE_CLUSTER_SIMILARITY` | `0.85` | fingerprint 類似度閾値 |
| `DSRE_HARMONIZE_METADATA` | `1` | 0 でメタデータ伝播を無効化 |
| `DSRE_SCORE_OPUS_BITRATE` | `182k` | 採点用 opus ビットレート |
| `DSRE_INTERACTIVE_CONFIRM` | `0` | 1 でフラグ群に UI ダイアログ (opt-in) |
| `DSRE_DISC_DIR_ALWAYS` | `0` | 1 で単 disc アルバムでも `Disc 01/` 作成 |
| `DSRE_SCORE_WEIGHT_DR` | `3.0` | スコア式 DR 係数 |
| `DSRE_SCORE_LUFS_TARGET` | `-14` | LUFS 目標値 |

---

## 19. リスク と 緩和策

| リスク | 緩和 |
|---|---|
| fingerprint 誤クラスタ (false positive) | 閾値 0.85 (高保守的) + 内容ハッシュ verify (同 cluster 内で content hash 完全一致 → 真の重複) |
| fingerprint 取りこぼし (false negative) | 単独 singleton として通常処理続行 (損失なし) |
| canonical 選定が間違う | 「既値尊重」ルールでユーザー意図保護 / 伝播ログ詳細記録 |
| version 系タグ消失 | 伝播対象外 リスト で絶対保護 |
| メタデータ書き換えで PICTURE 喪失 | mutagen の clear_pictures + add_picture 順序、Sub-A の知見流用 |
| fpcalc.exe 配布漏れ | spec で明示バンドル + 起動時存在確認 + ない場合は警告して fp 機能無効化 (単独 group 扱いで処理続行) |
| 並列採点で CPU 飽和 | max_workers 制御 + UI で進捗表示 |
| Windows 長パス | `\\?\` + 警告ログ |
| 重要バージョン誤判定で破棄 | 識別 rename + ゴミ箱で復元 / version 系タグ 7 種で過剰統合防止 |
| 「数値良好だが実聴劣化」検出漏れ | 5 種フラグ + opt-in 確認ダイアログ |

---

## 20. Out of Scope (将来 sub-project)

- **Sub-project C**: メタデータ自動取得 (VGMDB/MusicBrainz/iTunes/AppleMusic API)
  - 設定差異 (VGMDB: CV声優名、MusicBrainz: キャラ名) のため完全自動化は人間確認残り
  - DSRE 本体ではなく別ツール推奨
  - **本設計は「1 版だけタグ済」状態を想定** → C を実装すれば「0 版タグ済」も対応可能になる
- **foobar での opus 変換**: ユーザー従来通り
- **AcoustID API 連携**: chromaprint fingerprint を AcoustID (MusicBrainz) に投げて global lookup → メタデータ自動取得。Sub-C 範囲。

---

## 21. 設計上の核心ロジック (要旨)

ユーザーの実ワークフローを構造的に解決する 1 文:

> **「同曲は metadata でなく acoustic fingerprint で識別する。metadata は fingerprint cluster 内で 1 file から自動伝播する。これにより 30 file 中 10 file だけタグ取得すれば残り 20 file は DSRE が補完する。」**

これが v4 の構造的核心。残りの全 section はこの核心からの派生実装詳細。
