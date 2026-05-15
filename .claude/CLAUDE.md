# DSRE プロジェクト階層ルール

## 主対象
- **メインファイル**: `DSRE.py` (単一ファイル、約 700-800 行、PySide6 製 GUI + DSP)
- **設定ファイル**: なし (入出力は `INPUT_DIR` / `OUTPUT_DIR` 定数でハードコード: `C:\Audio\DSRE` / `C:\Audio\DSRE\Output`)
- **Python**: 3.11 のみ対応 (PySide6 は 3.11 に install 済み、`py -3.11` で検証)
- **出力フォーマット (v1.6 で確定)**: **FLAC 96kHz / PCM_24 固定**。
  - v1.5 で 192k 化したが、ユーザー主観で「ボーカル裏に高音乗り」違和感 + 計算 2 倍の負荷増 → v1.6 で 96k に revert
  - 192k は DSEE HX 思想 (Sony 公式も 96k 上限) を逸脱、可聴域外倍音が DAC/HP で intermodulation を生む仮説
  - v1.4 の 32bit float (foobar 測定で v1.3 と同値で revert) と並ぶ「過剰品質→副作用」事例の 2 つ目

## 由来・フォーク事情
- 本家: [x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer](https://github.com/x1aoqv/DSRE---Digital-Sound-Resolution-Enhancer) (507 行)
- このフォーク: [Rei3100/DSRE---Digital-Sound-Resolution-Enhancer](https://github.com/Rei3100/DSRE---Digital-Sound-Resolution-Enhancer) (ChatGPT で凝縮後、v1.4 で ~700 行)
- 参考 (別派生 v2.0): [Urabewe/DSRE v2.0 Enhanced](https://github.com/Urabewe/DSRE---Digital-Sound-Resolution-Enhancer-English) (2000 行超、UI 改善・負荷選択・リトライ等のアイデア参考)
- **DSRE 系 GitHub 実装の参考は上記 2 つ**。音響概念・信号処理理論・他分野の音質記事は制限なし (詳細は `~/.claude/CLAUDE.md` の「参考元の範囲」節)
- 音質は本家 zansei_impl の方向性を基準に、**劣化させない改善は積極的に採用**
- v2.0 の psychoacoustic_enhancer / multiband_exciter も**客観検証を通せば採用可能** (v1.4 以降)

## 絶対ルール (2026-04-25 改訂、v1.5 以降)
1. **UI 最低限の維持 + 改善歓迎**: ボタン名「開始」「一時停止」「取消」は維持、`setWindowTitle("DSRE")` は維持。負荷選択・システムトレイ・アイコン表示等の**機能拡張は可**。resize 値は内容に応じて調整可 (v1.4 は 340x210)
2. **音響処理は改善歓迎、ただし意味のある変更のみ**: `zansei_impl` / `freq_shift_mono/multi` / `safe_butter`(→`safe_butter_sos`) の**計算式を変えて良い**。条件は「劣化させない」+「**数値で差が出る (EQUIV のみは採用しない)**」。v1.4 の WAV 32bit float が foobar で v1.3 と同値だった事件を踏まえ、改善は**数値 + 主観の両方で確認**。selftest で `verdict=DEGRADED` または `verdict=EQUIV` のみなら commit しない (グローバル「意味のない変更禁止」ルール参照)
3. **本家影響ゼロ**: `git remote` は `origin` (Rei3100/DSRE) のみ。**`upstream` remote を絶対に追加しない**。本家への PR/push は永久禁止
4. **`run_hidden` は `CREATE_NO_WINDOW` 単独で使う** (STARTUPINFO は不要、yt-dlp GUI v17.2 と揃える)
5. **Windows subprocess**: 必ず `creationflags=CREATE_NO_WINDOW`、コマンドプロンプトをポップさせない
6. **INPUT_DIR / OUTPUT_DIR はハードコード維持** (個人ワークフローで固定、UI から変更させない方針)
7. **新機能は plan → 承認 → 実装** (重大破壊変更のみ。育成 9 項目テーブル義務は Phase 3 環境再設計で撤廃)
8. **出力フォーマット FLAC 96kHz / PCM_24 固定** (v1.6 確定)。**sr / bit / layer 数等の品質パラメータを上げる前に「3 問チェック (利益/不足/重量)」+ 主観 A/B 必要性自問の両方が必要**。32bit 化 (v1.4) と 192k 化 (v1.5) は数値テストでは検出できない副作用を持っていた

## 音響処理改善ルール (v1.5 改訂)
- ユーザーは最終の foobar2000 実聴確認のみ。**仮説・実装・客観検証は Claude が完結**
- 変更時に selftest で必ず以下を計測:
  - 旧実装との数値差 (max_abs_diff, rms_diff)
  - スペクトル差 (bin あたりの dB 差)
  - 波形ピーク移動量
  - 3 負荷レベルでの determinism (同一入力で同一出力)
  - **psychoacoustic A/B (v1.5 追加)**: DR / spectral centroid / high-freq rolloff / spectral flatness の 4 指標を旧→新で比較
- `verdict: EQUIV / IMPROVED / DEGRADED` を判定しログ出力。DEGRADED は CI で exit 1、EQUIV のみは warning
- 指標ライブラリ (pyloudnorm 等) を**社内検証に使うのは自由**、出荷物に含めるかは都度判断

## DSEE HX 系設計指針 (v1.5 以降、psychoacoustic 歓迎)

DSRE の音響処理は **DSEE HX 思想**を継承する (グローバル CLAUDE.md「DSRE 方向性記事」節参照):

- **DSEE HX 思想は「44.1k → 96k アップスケーリング」が核**、Sony 公式も 96k 上限
- 「ハイレゾ化」≠「アップサンプリング」。DSRE は後者であり 96k で十分 (192k は対象外)
- **段階的帯域拡張**: 低/中/高/超高域で別々に処理 (v1.5 で mid 3-8k / high 8-16k / ultra 16-24k に分割)
- **adaptive processing**: 入力 hf_ratio (4kHz 以上 / 総エネルギー) を測定し処理強度を動的調整
  - 圧縮音源 (hf_ratio < 0.05) → 強化 (layer 12/8/6)
  - 中間 → 標準 (layer 8/6/4)
  - 高域リッチ (hf_ratio ≥ 0.2) → 抑制 (layer 4/3/2)
- **plausible な高域推定**: 既存 freq_shift (single-sideband) + 帯域内 spectral centroid 保持
- **採用条件**: psychoacoustic A/B で IMPROVED 判定 (DR ±0.5dB 以内 + centroid 上昇 + rolloff 上昇)
- **multiband_exciter / psychoacoustic_enhancer 系の改造**は自由。**「劣化しない」+「数値で差が出る」**を満たせば採用

## 定数中央集約 (ファイル冒頭)
- `INPUT_DIR` / `OUTPUT_DIR`
- `HARMONIC_LAYERS` / `HARMONIC_DECAY` / `PRE_HP_CUTOFF_HZ` / `POST_HP_CUTOFF_HZ` / `TARGET_SR` / `FILTER_ORDER`
- `@dataclass(frozen=True) class DSREParams` + `PARAMS = DSREParams()` インスタンス

音響パラメータを変更する場合は、上記定数 + DSREParams の両方を更新する。

## 依存管理
- **ランタイム**: `requirements.txt` (PySide6/numpy/scipy/librosa/resampy/soundfile/send2trash のみ、UTF-8)
- **ビルド**: `requirements-dev.txt` (pyinstaller 系のみ)
- librosa が引きずる audioread/numba/llvmlite/pooch/soxr/joblib 等は明示列挙しない (pip に任せる)

## Claude Code 運用
- 編集は Claude Code 側で完結、ユーザーは GitHub 運用の知識ゼロ前提
- push は**ユーザーが「push して」と明示したとき**のみ
- コミットはオプション単位で分割 (`refactor:` / `fix:` / `deps:` / `tooling:` / `build:` プレフィックス)
- 編集後の自動検証 (py_compile) は `~/.claude/settings.json` の PostToolUse hook が走る (yt_dlp_gui.py と DSRE.py の両方が対象)

## ビルド運用 (numpy 未検出事件 2026-04-20 以降)
- **ビルドは `build.ps1` (ローカル) か GitHub Actions の 2 経路のみ**、素の `pyinstaller DSRE.py` 直叩きは禁止 (numpy 2.x を取りこぼす)
- **`DSRE.spec` を必ず使う**: `collect_all` で numpy/scipy/librosa/numba/llvmlite/resampy/soundfile/send2trash を丸ごとバンドル
- **Python 3.11 固定**: 3.10 等でビルドしない (ローカル env と ABI 一致)
- **依存ロック**: `pyinstaller==6.15.0` + `pyinstaller-hooks-contrib==2025.8` (requirements-dev.txt)、勝手にアップデートしない
- **CI スモークテスト必須**: build.yml に `_internal/numpy` `_internal/scipy` `_internal/librosa` の存在確認、落ちたら artifact 作らない
- **新しい依存ライブラリ追加時**: DSRE.spec の `for mod in (...)` に追記し、`.github/workflows/build.yml` のスモーク配列にも追記

## 自動デプロイレール (ユーザー操作ゼロ)
Claude が DSRE 関連を編集したら、以下を自走で完遂:
1. commit & push (origin/main) → GitHub Actions が `on: push` で自動起動
2. `gh run watch --exit-status` で完了待機 (長時間は `ScheduleWakeup` 270s ポーリング)
3. `gh run download --name DSRE_private` で `DSRE_private.zip` 取得
4. `deploy.ps1 -ZipPath <zip>` を呼ぶ
   - 既存 `DSRE.exe` プロセス停止
   - `C:\FreeSoft\DSRE` → `C:\FreeSoft\DSRE.bak` バックアップ
   - zip 展開 → `C:\FreeSoft\DSRE` 配置
   - スモーク (exe + numpy + scipy + librosa 同梱確認)
   - `DSRE.exe` 起動
5. ユーザーには「何を直した」「CI 結果」「起動完了」を 3 行で報告
ロールバック: `Rename-Item C:\FreeSoft\DSRE.bak C:\FreeSoft\DSRE` (現用を先に rm してから)

## 関連サブエージェント
- `dsre-specialist` (DSP 勘所・本家差分・音質改変禁止ルール熟知)
- `code-reviewer` / `bug-hunter` / `refactor-finder` は汎用として使える
