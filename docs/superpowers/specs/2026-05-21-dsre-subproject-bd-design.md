# DSRE Sub-project B+D: ワークフロー自動化 + 客観品質ゲート 設計書 (v3)

**Date:** 2026-05-21 (v3 全面再点検)
**Scope:** 重複検出 / 客観品質スコアリング (opus 採点) / 最良選択 / pre 処理仕分け+リネーム / post 処理仕分け+リネーム / 識別可能な破棄 / 同曲別バージョン間メタデータ整合 / 冪等性 / 失敗リジューム
**Integrated:** B (ワークフロー) と D (客観品質ゲート) を統合
**Out of scope:** メタデータ自動取得 (Sub-project C・将来別ツール)、foobar での opus 変換 (DSRE の役割ではない)

---

## 0. 出力フォーマット

| 種類 | 形式 | 役割 |
|---|---|---|
| **最終出力** | **FLAC 96kHz PCM_24** | DSRE 処理結果。既存通り。これが残る。 |
| **opus** | **採点用テンポラリ** | 同条件比較のためだけに生成、解析後即削除 |

opus は DSRE の最終出力ではない。foobar での opus 変換は **ユーザーが従来通り別途行う**。

---

## 1. ユーザーの既存ワークフロー (前提)

```
音源 (raw FLAC、同曲複数バージョンあり)
   ↓ mp3tag でメタデータ取得 (手動、将来 Sub-project C で自動化候補)
   ↓ DSRE 用フォルダに処理前として出力 ← Sub-project B+D が自動化
   ↓ DSRE 処理
   ↓ アートワーク埋め込み (Sub-project A で対応済)
   ↓ foobar 互換階層に仕分け+リネーム ← Sub-project B+D が自動化
   ↓ foobar で opus 変換 (DSRE 外、ユーザーが従来通り)
```

思想 (原指示): **出来るだけ楽に・可能な部分は全部自動化・手動確認とかかかる時間を極限まで減らす**。

---

## 2. 全体パイプライン

```
[STAGE 1: 処理前段階]
  INPUT_DIR スキャン (.flac、再帰)
    ↓ 既処理 (dsre_version タグあり) は冪等 skip
  各ファイルのメタデータ抽出 (mutagen)
    ↓
  同曲グルーピング (5-key match + 別バージョンタグ差分検出)
    ↓
  別バージョン間のメタデータ整合 (共通フィールドを最良候補値で統一)
    ↓
  グループ内の各ファイルを採点 (opus 182k VBR エンコード → デコード → 解析、並列)
    ↓
  グループ内で最良 1 つを選択 (要注意フラグ判定込み)
    ↓
  非最良は識別可能リネーム + send2trash
    ↓
  最良 1 つを foobar 互換階層で INPUT_DIR 内に再配置 (rename + sort)

[STAGE 2: DSRE 処理]
  最良ファイル群を順次 DSRE 処理 (zansei_impl、既存)
    ↓ アートワーク埋め込み (Sub-A、既存)
  96kHz FLAC PCM_24 出力

[STAGE 3: 処理後段階]
  出力 FLAC を foobar 互換階層で OUTPUT_DIR 内に配置 (同テンプレート)
  → INPUT_DIR と OUTPUT_DIR は 1:1 同型のレイアウトになる
```

---

## 3. 同曲グルーピング

### 3.1 マッチキー (5-key)

```python
key = (
    norm(artist),
    norm(album),
    discnumber or "1",
    tracknumber.zfill(3),
    norm(title),
)
```

`norm`: lowercase + 全角半角統一 + 連続空白 → 単一 + 前後 trim。

完全一致した複数ファイルが「同曲候補」になる。

注: `albumartist` は使わない (ユーザーの実環境タグに存在しない)。`artist` のみ。

### 3.2 別バージョン判定 (バージョン系タグ差分)

同 key の候補グループ内で以下のタグのいずれかが異なれば **別バージョン** として独立保持:

| タグ名 | 用途 |
|---|---|
| `version_info` | バージョン汎用 |
| `cover_type` | カバー種別 |
| `live_type` | ライブ種別 |
| `vocal_type` | ボーカル種別 |
| `remaster_info` | リマスター情報 |
| `arrange_type` | アレンジ種別 |
| `m_number` | M ナンバー |

(ユーザー原テンプレートで `[..]` suffix として展開される 7 タグ。タグの有無・値の差分で別版判定。)

これら全部同一 (または全部空) → 真の重複として 1 つに絞る。
いずれか異なる → 別バージョン、別 group として両方残す。

正規表現で `[Live]` 等のタイトル末尾を検出する手法は採らない (発明禁止)。

### 3.3 同曲別バージョン間のメタデータ整合

ユーザー思想 「全部同じメタデータじゃないとなのよ」 (同曲は共通メタデータ揃え) に従い、別バージョンとして保持する複数 file に対し:

- **共通フィールド** (artist, album, discnumber, tracknumber, title, date, genre, age, circle, category, source, grouping, franchises, products, series, brand, subtitle, elements, project, collaboration, group, unit, album_type): グループ内で値が異なるなら **最良候補の値に統一** (mutagen で書き換え)
- **バージョン系フィールド** (§3.2 の 7 タグ): 触らない (それぞれの版独自)
- **featuring/produced**: グループ内全 file で同一なら統一、異なれば触らない (版ごとに異なる場合あり)

これにより、別版 file 群は「version 系タグ以外は完全同一」状態になり、後のリネーム・仕分け先パスが「version suffix だけ違って同一階層」に揃う。

### 3.4 メタデータ欠落

artist / album / title 等の主要タグ欠落:
- 同 key 計算不可 → 単独 group 扱い (グルーピング対象外)
- 警告ログ
- DSRE 処理は通常通り進める (foobar 仕分けは可能な範囲で適用、欠落レベルは省略)

---

## 4. 客観品質スコアリング (B+D の核)

### 4.1 採点用 opus エンコード

音量正規化は **DSRE 既存の `save_flac24_out` 内の volume normalization をそのまま流用** (true peak 8x oversampling → -0.3 dBFS target、clip=0 を構造的保証する既存ロジック)。新規に ReplayGain 相当を実装しない。

```python
# Step 1: 既存 DSRE 音量正規化を流用 (save_flac24_out 内の関数を抽出)
audio_norm = dsre_normalize_volume_for_clip_zero(audio, sr)

# Step 2: tmp FLAC 経由で ffmpeg libopus 182k VBR (ユーザー既存設定と同)
ffmpeg -i {tmp_norm.flac} \
  -c:a libopus -b:a 182k -vbr on -application audio \
  {tmp.opus} -y

# Step 3: opus デコード → MetricsComputer.compute()
# Step 4: スコア計算
# Step 5: tmp opus / tmp norm flac を try/finally で必ず削除
```

DSRE と同じ正規化を使う = 「DSRE 処理後 opus 変換した時の最終形」を限りなく忠実にシミュレート。フェア比較の本質を満たす。

### 4.2 並列化

opus エンコードは subprocess、CPU 並列可能。`ThreadPoolExecutor(max_workers=os.cpu_count())` で並列化。50 ファイルで 4-8 分 → 1-2 分。

中断対応: 各 future の tmp は try/finally で削除、cancel 時も漏れなし。

### 4.3 スコア式 (初期値、調整可能)

opus デコード後の解析値を以下で合算 (0-100):

```python
score = 0.0
score += clamp(dr, 0, 20) * 3.0           # DR (max 60)
score += clamp(20 - abs(lufs + 14), 0, 20) # LUFS proximity (max 20)
score += clamp(hf_ratio_12k, 0, 0.05) * 200 # HF (max 10)
score += clamp(flatness, 0, 0.5) * 20      # spectral flatness (max 10)
score -= max(0, 6 - plr) * 2                # PLR < 6 ペナルティ (max -12)
score = clamp(score, 0, 100)
```

clip_count / peak_db は normalize で揃うため減点対象外。
重み・閾値は env var で上書き可能 (§16)。実運用で観測しながら調整。

### 4.4 要注意フラグ (D: 数値良好でも実聴劣化を捕まえる)

過去問題: 「数値は良いのに聴いたら明らかに劣化」のパターン検出フラグ:

| フラグ条件 | 検出意図 |
|---|---|
| `hf_ratio_16k < 0.0005` かつ `rolloff_hz < 17000` | 高域カット (lossy origin の可能性) |
| `flatness < 0.05` | スペクトル平坦性極小 (brick wall master) |
| `dr < 6` | DR 危険水準 (ハイパー圧縮原音) |
| `harmonic_1k_proxy > 0.5` かつ `dr > 12` | 高 DR だが歪み多 (アーティファクト疑い) |
| `hf_ratio_8k > 0.3` かつ `centroid_hz > 6000` | 不自然な高域偏重 (人工的アップサンプル疑い) |

フラグは ログ + UI の summary に表示。デフォルトは **確認なしで自動進行** (§5)。

### 4.5 自動選択ロジック

```python
def select_best(group: list[FileScore]) -> Selection:
    no_flag = [f for f in group if not f.flagged]
    pool = no_flag if no_flag else group
    best = max(pool, key=lambda f: (f.score, f.file_size))
    return Selection(best, all_flagged=(not no_flag))
```

同点: ファイルサイズ大 (情報量多) を優先。

---

## 5. ユーザー介入の最小化 (思想実装)

「手動確認とかかかる時間を極限まで減らす」に従い:

- **デフォルト**: 全 group 自動進行 (フラグ付きグループも警告ログ出すだけで処理続行)
- **env `DSRE_INTERACTIVE_CONFIRM=1`** で初めて UI ダイアログを有効化 (opt-in)
- ユーザーは事後に `OUTPUT_DIR/_workflow_log.txt` を見て判断 → 必要に応じてゴミ箱から復元

これにより、処理を投げた後の手動介入が原則ゼロになる。

---

## 6. foobar 互換フォルダ階層

### 6.1 ユーザー原テンプレート (再掲)

ユーザー指示のフルテンプレ:
```
Audio / [{device}] / [{genre}] / [{age}] / [{circle}] / [{category}] /
[{source}] / [{grouping}] /
[{franchises if franchises != project}] /
[{products if products != project and != brand}] /
[{series if series != project}] /
[{brand}] /
[{subtitle if subtitle != brand and != franchises}] /
[{elements if elements != brand and != franchises and != subtitle}] /
[{project}] / [{collaboration}] / [{group}] / [{unit}] / [{album_type}] /
[{year}/{month}/] /
{album} [(date)] /
Disc {discnumber:02d} /
{discnumber:02d}.{tracknumber:03d}.{title}
  [- {artist}|- {artist} [feat. {featuring}]]
  [[Prod. {produced}]]
  [[{arrange_type}]]
  [[{version_info}]]
  [[{remaster_info}]]
  [[{cover_type}]]
  [[{live_type}]]
  [[{vocal_type}]]
  [[{m_number}]]
.flac
```

各レベル/サフィックスは **タグ存在で挿入、欠落で省略**。

### 6.2 DSRE での適用 (OUTPUT_DIR ＝ `C:\Audio\DSRE\Output\`)

DSRE 側は `Audio/[{device}]/` を **使わない** (OUTPUT_DIR の `Audio` と重複)。
DSRE テンプレート:
```
{OUTPUT_DIR}/
  [{genre}/] [{age}/] [{circle}/] [{category}/] [{source}/] [{grouping}/]
  [{franchises}/] [{products}/] [{series}/] [{brand}/]
  [{subtitle}/] [{elements}/] [{project}/] [{collaboration}/]
  [{group}/] [{unit}/] [{album_type}/]
  [{year}/{month}/]
  {album}[' ('date')'] /
  [Disc {disc:02d}/]      ← 単 disc アルバムは省略
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

実装は `FoobarPathBuilder` クラス (純関数集合) に閉じ込め。

### 6.3 複数値タグの delimiter 処理

foobar 原テンプレートで `/` を空白等に置換しているのに従う:
- artist の `/` → `_ ` (アンダースコア + 空白)
- featuring の `/` → `* ` (アスタリスク + 空白)
- arrange_type, version_info, remaster_info の `;` → ` /` (空白スラッシュ)

### 6.4 衝突回避

同パス既存:
- 内容ハッシュ一致 → skip + ログ
- 内容相違 → 数字 suffix `_2.flac`, `_3.flac`, ...

### 6.5 Windows パス制約

- 禁止文字 `< > : " / \ | ? *` を全角に置換
- 末尾の `.` や空白を削除
- パス長が 250 chars 超過時: `\\?\` プレフィックス使用 (Windows API で MAX_PATH 回避)
- 250 chars 超過は警告ログ (将来テンプレ縮約検討)

### 6.6 単 disc アルバムの Disc 階層省略

album 内の全 track の discnumber が "1" のみ → `Disc 01/` 階層を省略。
複数 disc の album → 全 track に `Disc NN/` 階層を作成 (混在防止)。

---

## 7. 処理前 (STAGE 1) の仕分け+リネーム

最良ファイル選択後、それを **§6.2 の foobar テンプレートで INPUT_DIR 内に再配置** (rename + sort)。

これにより:
- ユーザーが INPUT_DIR を覗いた瞬間、処理予定 file 群が整理された階層で見える
- STAGE 3 後の OUTPUT_DIR と完全同型 (1:1 対応)
- DSRE 処理は再配置後のパスから入力

非最良 (重複) は §8 へ。

---

## 8. 破棄処理 (識別可能リネーム → ゴミ箱)

### 8.1 リネーム規則

非最良ファイルを send2trash する前に:
```
{元のファイル名}.__discarded_dsre_inferior__.flac
```

例: `track01_v2.flac` → `track01_v2.__discarded_dsre_inferior__.flac` → trash 行き。
ユーザーがゴミ箱を覗くと「DSRE が劣品質と判定して捨てた」と一目で識別可能。

### 8.2 破棄ログ

`OUTPUT_DIR/_workflow_log.txt` に追記:
```
[2026-05-21T18:00:00] group=Artist-Album-Disc01-Track03
  kept    : v3.flac     score=78 (clean)
  discard : v1.flac     score=45 flags=brick_wall
  discard : v2.flac     score=62 (clean) — lower score
```

---

## 9. 処理後 (STAGE 3) の仕分け+リネーム

DSRE 処理結果の FLAC を **§6.2 の同テンプレートで** OUTPUT_DIR 内に配置。

INPUT_DIR レイアウト ↔ OUTPUT_DIR レイアウトが 1:1 同型になる。foobar から OUTPUT_DIR を読めば既存設定のまま opus 変換可能。

---

## 10. 冪等性 (再実行安全性)

スキャン時に各 file の `dsre_version` Vorbis Comment タグを確認:
- タグ存在 → 既処理、skip (再 DSRE しない、再仕分けもしない)
- タグ未在 → 通常処理

これによりユーザーが何度 「開始」 ボタンを押しても破壊的変更は発生しない。

---

## 11. 失敗時のリジューム

journal は持たない。状態は INPUT_DIR / OUTPUT_DIR のファイル配置とタグで自己記述する:

- STAGE 1 中で停止: 仕分け済の最良ファイルは INPUT_DIR の階層位置で識別可能、未処理は残置。再実行時、`dsre_version` 無しの file を再スキャン。
- STAGE 2 中で停止: 最良ファイル群は INPUT_DIR にあり、一部が処理済。再実行で未処理分のみ DSRE 処理。
- STAGE 3 中で停止: DSRE 完了 file は OUTPUT_DIR にあり、仕分け前のものは OUTPUT_DIR ルートに残る。再実行で残置分を仕分けする (= STAGE 3 のみリトライ可能)。

冪等性 (§10) と組み合わせて、何度落ちても再走で続きから自動的に進む。

---

## 12. 処理順序 (バッチ順)

採点・処理ともに以下の順:
```
ORDER BY artist, album, discnumber, tracknumber
```

I/O ローカリティ・進捗の自然さ。

---

## 13. UI 変更 (MainWindow、既存維持 + 進捗テキスト改善)

「常に全自動」なので UI は最小限維持。

進捗テキスト形式:
```
[STAGE 1] スキャン 45 files (3 既処理 skip)
[STAGE 1] グルーピング 28 group (8 重複)
[STAGE 1] 採点 並列 12/45 (45 sec ETA)
[STAGE 1] 最良選択 完了、破棄 17
[STAGE 1] 整列 INPUT_DIR
[STAGE 2] DSRE 処理 [3/28] {album}/{track}
[STAGE 3] 整列 OUTPUT_DIR
完了 28/28 (要確認 2 → _workflow_log.txt 参照)
```

トレイメニュー: 既存維持 (Sub-A で簡素化済の「開始」のみ)。

要確認ダイアログは **デフォルト OFF** (§5)。`DSRE_INTERACTIVE_CONFIRM=1` で opt-in。

---

## 14. 新規モジュール

| クラス | 責務 | 既存依存 |
|---|---|---|
| `MetadataExtractor` | mutagen で原テンプレ参照タグ群を抽出 | mutagen |
| `TrackGrouper` | 5-key match + version タグ差分 | (純関数) |
| `MetadataHarmonizer` | グループ内共通フィールドを最良値に統一 | mutagen |
| `QualityProbe` | DSRE 流用 normalize + opus encode + decode + 解析 | ffmpeg, MetricsComputer, DSRE 既存 |
| `BestSelector` | グループ内最良選択 (フラグ判定込み) | (純関数) |
| `DiscardHandler` | 識別可能リネーム + send2trash | send2trash |
| `FoobarPathBuilder` | テンプレ展開 (optional level / suffix、サニタイズ、長パス対応) | (純関数) |
| `WorkflowOrchestrator` | STAGE 1-3 統括、冪等性 + リジューム制御 | 全部 |

既存 `Worker._process_one` (DSRE 本体処理) は STAGE 2 から呼ばれる関数として温存。`WorkflowOrchestrator` が `Worker.run` ループを置き換える。

DSRE 既存 `save_flac24_out` 内の volume normalization 部分を独立関数として抽出して採点用にも流用可能にする (リファクタ、機能変更なし)。

---

## 15. 依存追加

opus エンコードは同梱 ffmpeg + libopus。新規バイナリ依存なし。
Python 側も追加なし (mutagen は Sub-A で導入済)。

---

## 16. 変更ファイル

| ファイル | 変更種別 | 概要 |
|---|---|---|
| `DSRE.py` | 大幅追加 + 軽微リファクタ | 8 新クラス、Worker 改修、Orchestrator 統括、`save_flac24_out` 内の volume norm 関数を抽出 |
| `requirements.txt` | 変更なし | - |
| `DSRE.spec` | 変更なし | - |

---

## 17. テスト計画

| 検証項目 | 方法 | 合格基準 |
|---|---|---|
| グルーピング 5-key | mock 3 同曲 | 1 group |
| グルーピング 別バージョンタグ差分 | `version_info` 違い | 別 group |
| グルーピング メタ欠落 | title 欠落 | 単独 group |
| メタデータ整合 | グループ内 genre 不一致 | 最良値で統一 |
| 採点 normalize | DSRE 同等 norm | clip=0 達成 |
| 採点 opus エンコード+削除 | 通常 + 中断 | tmp 漏れなし |
| 採点 スコア | 高品質 mock | score > 80 |
| フラグ HF cliff | rolloff=15k hf_16k=0 | flagged |
| 最良選択 全 clean | 3 件 | max score |
| 最良選択 全 flagged | 3 件 | max + all_flagged=True |
| 破棄リネーム | non-best | `__discarded_dsre_inferior__` suffix |
| ゴミ箱投入 | discard | send2trash 成功 |
| FoobarPath all tags | full metadata | テンプレ完全展開 |
| FoobarPath missing | genre のみ | genre 階層省略 |
| FoobarPath sanitize | `?` 入り | 全角化 |
| FoobarPath 長パス | >250 chars | `\\?\` プレフィックス |
| FoobarPath delimiter | artist に `/` | `_ ` 置換 |
| 単 disc 省略 | 全 disc=1 | Disc 階層なし |
| 衝突回避 | 同パス | `_2.flac` |
| 冪等性 | 既処理 file | skip |
| リジューム STAGE 2 中断 | mock 中断 + 再実行 | 続き処理 |
| STAGE 1→2→3 統合 | mock 5 group | 全 STAGE 完了 + 同型レイアウト |
| selftest | `--selftest` | verdict=EQUIV |

---

## 18. 設定可能パラメータ (env var)

| env var | デフォルト | 説明 |
|---|---|---|
| `DSRE_WORKFLOW` | `1` | 0 で旧パイプライン (Sub-A 状態) に強制 fallback |
| `DSRE_PRESORT_INPUT` | `1` | 0 で STAGE 1 の INPUT_DIR 仕分けをスキップ |
| `DSRE_SCORE_OPUS_BITRATE` | `182k` | 採点用 opus ビットレート (foobar 既存設定と同) |
| `DSRE_INTERACTIVE_CONFIRM` | `0` | 1 でフラグ群に UI 確認ダイアログ (default OFF = 全自動) |
| `DSRE_HARMONIZE_METADATA` | `1` | 0 で別バージョン間メタデータ整合を無効化 |
| `DSRE_DISC_DIR_ALWAYS` | `0` | 1 で単 disc アルバムでも `Disc 01/` 階層を作る |
| `DSRE_SCORE_WEIGHT_DR` | `3.0` | スコア式 DR 係数 |
| `DSRE_SCORE_WEIGHT_LUFS_TARGET` | `-14` | LUFS 目標値 |

---

## 19. リスク と 緩和策

| リスク | 緩和 |
|---|---|
| 重要バージョン誤判定で破棄 | (1) 識別可能リネーム+ゴミ箱で復元容易 (2) ログ詳細記録 (3) 別バージョンタグ判定で過剰統合防止 |
| メタデータ整合で意図しないタグ上書き | バージョン系 7 タグは絶対に触らない / featuring は完全同一時のみ統一 |
| 採点 normalize が DSRE 内部と乖離 | `save_flac24_out` 内の関数を抽出して共有 (重複実装しない) |
| opus 採点 CPU 重 | 並列化 + tmp file try/finally |
| パス長超過 | `\\?\` プレフィックス + 警告 |
| 「数値良好だが実聴劣化」検出漏れ | フラグ 5 種 + 任意の手動再確認ルート (`DSRE_INTERACTIVE_CONFIRM=1`) |

---

## 20. Out of Scope (将来 sub-project)

- **Sub-project C**: メタデータ自動取得 (VGMDB/MusicBrainz/iTunes/AppleMusic API)
  - VGMDB: artist=CV声優名 → artist タグ off
  - MusicBrainz: artist=キャラ名 → artist タグ on
  - 設定の差異が大きく、人間確認ステップは残る見込み
  - DSRE 本体ではなく別ツールとして検討
- **foobar での opus 変換**: ユーザーが従来通り別途実施。
- **acoustic fingerprint (chromaprint)**: メタデータ欠落時の最終手段、現状不要
