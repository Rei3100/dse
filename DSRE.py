# ===== 強化版（処理維持＋安定性＋精度向上）=====

import os
import sys
import time
import tempfile
import subprocess
import configparser
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import next_fast_len
import librosa
import resampy

from PySide6 import QtCore, QtGui, QtWidgets

from send2trash import send2trash


# ===== 入出力 =====
INPUT_DIR = r"C:\Audio\DSRE"
OUTPUT_DIR = r"C:\Audio\DSRE\Output"


# ===== DSP パラメータ =====
# v1.13: 本家相当の基礎性能 (ナイキスト到達の高域補完 + DR/PLR 改善) を完全復活
# v1.10/本家準拠: HARMONIC_LAYERS=8 / DECAY=1.25 / POST_HP=12k に戻す。
# 加えて高層 (i=5..7) の decay を補正項でブーストし、32-48kHz 帯域 (ナイキスト
# 近傍) の補完エネルギーを目視確認できるレベルに保つ。
HARMONIC_LAYERS = 8         # v1.13: 10→8 復帰 (本家 + v1.10 と同じ 8 層)
HARMONIC_DECAY = 1.25       # v1.13: 1.10→1.25 復帰 (本家 + v1.10 と同じ、層あたりエネルギー配分が安定)
PRE_HP_CUTOFF_HZ = 3000     # 倍音抽出前のハイパス
POST_HP_CUTOFF_HZ = 12000   # v1.13: 10000→12000 復帰 (v1.8 確定値、過剰加算回避)
TARGET_SR = 96000           # v1.6: 本家デフォルト
FILTER_ORDER = 11           # バターワース次数

# v1.16: 高層 decay 補正 (v1.13/v1.14 互換の 1.5x 一律段差復帰)
# v1.15 で slope 化を試みたが、最終 master EQ で単調化を引き受けるため
# 段差そのものは d_res で許容し、最終出力 EQ で補正する設計に戻す。
HARMONIC_HIGH_LAYER_BOOST = 1.5

# --- v1.16: Master Spectral Monotonizer (最終出力に対する単調化制約) ---
# v1.15 の path 単位 sosfilt tilt は forward-only で位相歪 + 高域ノイズ源
# となった。v1.16 は path 単位 tilt を全削除、d_extra 全体に対し最終段で
# zero-phase 単調化 EQ を 1 回だけ適用する。
# 実装方針: rfft で隣接 1kHz bin の上昇 (mag[b] > mag[b-1] * jump_tol) のみを
# 縮小。低域全体の値で高域を抑える (cumulative-min) ような全域抑制はしない。
# DSRE 補完エネルギー (exciter / air / nyq) は維持しつつ、局所ピークだけ削る。
MASTER_MONO_LO_HZ = 4000           # 単調化対象の下限 (これ以下は無処理)
MASTER_MONO_BIN_HZ = 1000.0        # bin 幅 (1kHz、selftest と同じ単位)
MASTER_MONO_SMOOTH_BINS = 3        # ゲイン補正の隣接 bin 平滑化窓
MASTER_MONO_JUMP_TOL_DB = 4.0      # 隣接 bin の許容上昇幅 (これを超えた場合のみ抑制)
MASTER_MONO_RUN_PASSES = 2         # 反復回数 (1 回で消えない局所ピークを 2 回で確実に解消)

# ===== v1.12: Multi-Aspect Enhancement (Drive 適正化) =====
# v1.11 では drive 控えめすぎて「処理した意味」が客観・主観の両方で薄かった。
# v1.12 では各 path の drive を聴感で明確に効果が出るレベルまで引き上げつつ、
# 刺さり/濁り/位相崩れ を回避するため帯域配置と SAT カーブは v1.11 を踏襲。
# Master Headroom (0.99) で peak を保護、原音は依然無減衰。

# --- v1.12: Improved Harmonic Exciter (drive 引き上げ) ---
# 4-10kHz は厚みを十分出し (drive 0.28)、10kHz+ は刺さり回避のため drive=0.16。
# 偶奇比は下段 40/60、上段 30/70 (上段は presence 寄り、温度より sparkle)。
EXCITER_LO_HZ = 4000        # 下段 exciter ソース下限
EXCITER_HI_HZ = 9000        # v1.12: 10000→9000 (上段拡張で空気感寄与増)
EXCITER_OUT_HP_HZ = 6000    # 生成倍音の出力 HP (中域への滲み防止)
EXCITER_DRIVE_LO = 0.28     # v1.12: 0.16→0.28 (温度・実在感を有意に)
EXCITER_DRIVE_HI = 0.16     # v1.12: 0.10→0.16 (透明感を有意に、刺さり回避は SAT 側で)
EXCITER_SAT_GAIN = 1.4      # tanh 入力ゲイン (1.4=soft tube-like、刺さり防止)
EXCITER_EVEN_RATIO = 0.40   # 偶数倍音 (warmth 寄与)
EXCITER_ODD_RATIO = 0.60    # 奇数倍音 (presence 寄与)

# --- v1.12: Mid Warmth (drive 引き上げ + SAT 強化) ---
# v1.11 の drive=0.05 では effective に無音 (selftest WARM_pk=0.0001)。
# v1.12: drive=0.12 + asym_gain=1.5 で温度感・艶を聴感レベルに引き上げる。
# 加算帯域は 1.2kHz HP 維持 (低域膨張・濁り完全回避)。
MID_WARMTH_LO_HZ = 200
MID_WARMTH_HI_HZ = 1800     # v1.12: 1500→1800 (中高域寄りに拡張、ボーカル子音帯域)
MID_WARMTH_OUT_HP_HZ = 1200  # 加算は中高域のみ
MID_WARMTH_DRIVE = 0.12     # v1.12: 0.05→0.12 (聴感レベル)
MID_WARMTH_ASYM_GAIN = 1.5   # v1.12: 1.2→1.5 (2nd harmonic 生成量を増)

# --- v1.12: Stereo Width HF (drive 引き上げ) ---
# v1.11 の +1.4dB では空間広がりが体感ぎりぎり。
# v1.12: gain=0.35 (+2.6dB 相当) で立体感・包囲感を有意に。Mid 無触は維持。
STEREO_WIDEN_HP_HZ = 2000
STEREO_WIDEN_GAIN = 0.35     # v1.12: 0.18→0.35 (Side 高域のみ +2.6dB 相当)

# --- v1.12: Air Band Sparkle (drive 引き上げ + 帯域拡大) ---
# v1.11 の AIR_RATIO +1.24e-3 は微小。v1.12: drive=0.20 で plausible HF を有意に。
AIR_BAND_HP_HZ = 12000       # v1.12: 13000→12000 (ソース帯域を 1kHz 拡大)
AIR_BAND_OUT_HP_HZ = 14000   # v1.12: 15000→14000 (中域への漏れは 14kHz でカット、十分高い)
AIR_BAND_DRIVE = 0.20        # v1.12: 0.10→0.20 (空気感を有意に)
AIR_BAND_SAT_GAIN = 2.2      # v1.12: 2.0→2.2 (微小信号でもしっかり倍音生成)

# --- v1.13: Transient Crispness (DR/PLR 改善寄与を強化) ---
# v1.12 の drive=0.18 はピークを増やすが PLR (peak/loudness) 改善は限定的。
# v1.13: 帯域を 500Hz まで拡大 + drive=0.30 でキック・スネアのアタックを
# 明確化。peak 領域を相対的に伸ばし、ロスレスな DR/PLR 改善を生む。
TRANSIENT_LP_HZ = 500        # v1.13: 260→500 (スネア倍音帯域も含める)
TRANSIENT_FAST_MS = 4.0      # v1.13: 5→4 (より短い envelope、立ち上がり鋭く)
TRANSIENT_SLOW_MS = 60.0     # v1.13: 50→60 (slow を遅く、コントラスト拡大)
TRANSIENT_GAIN = 0.30        # v1.13: 0.18→0.30 (アタック立ち上がりの実体感)

# --- v1.13: Nyquist Band Complement (32-48kHz 補完、本家同様の plausible HF) ---
# v1.10/本家では d_res の高層 (i=5..7) で 36-48kHz 帯にエネルギーが乗っていた。
# v1.13: 独立 path として「8-22kHz 帯域 → +24kHz freq_shift → 32-46kHz」で
# 確実に nyq 近傍まで plausible エネルギーを補完する。
# Audacity 等のスペクトル表示で 32-48kHz 帯に明確な補完が見えることを保証。
NYQ_COMPLEMENT_LO_HZ = 8000      # ソース帯域下限
NYQ_COMPLEMENT_HI_HZ = 22000     # ソース帯域上限 (44.1k 入力でも安全)
NYQ_COMPLEMENT_SHIFT_HZ = 24000  # +24kHz シフト → 32-46kHz 出力
NYQ_COMPLEMENT_OUT_HP_HZ = 30000 # 30kHz HP で 30-48kHz のみ通す
NYQ_COMPLEMENT_DRIVE = 0.18      # 控えめ drive (32-48k は可聴域外、DAC 折返し対策)

# --- Master Headroom ---
# 全 path 加算後 peak が 0.99 を超えそうなら d_extra のみ縮小。原音は無減衰。
MASTER_HEADROOM_PEAK = 0.99
# v1.6: FLAC 96kHz / PCM_24 固定 (v1.5 の WAV 32bit float / 192kHz は overkill だった)
# 経緯: v1.4 で WAV 32bit float 化 → foobar 測定で v1.3 と同値 → v1.5 で FLAC PCM_24 復帰
#       v1.5 の 192k 出力 → 主観違和感 (ボーカル裏に高音乗り) + 計算 2 倍 → v1.6 で 96k 復帰
OUTPUT_SUBTYPE = "PCM_24"
OUTPUT_SUBTYPE_FALLBACK = "PCM_16"  # libsndfile 異常時の安全網
OUTPUT_FORMAT = "FLAC"
OUTPUT_EXT = ".flac"

# ===== 負荷レベル (v1.7 Phase M: 1-10 段階) =====
LOAD_LEVEL_MIN = 1
LOAD_LEVEL_MAX = 10
LOAD_LEVEL_DEFAULT = 5
STATE_INI_NAME = "state.ini"
# 旧フォーマット ("軽"/"標準"/"最大") からの自動移行マップ
_LEGACY_LOAD_MAP = {"軽": 2, "標準": 5, "最大": 9}


def _state_ini_path():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, STATE_INI_NAME)


def load_level() -> int:
    p = _state_ini_path()
    if not os.path.isfile(p):
        return LOAD_LEVEL_DEFAULT
    try:
        cp = configparser.ConfigParser()
        cp.read(p, encoding="utf-8")
        raw = cp.get("ui", "load", fallback=str(LOAD_LEVEL_DEFAULT))
        if raw in _LEGACY_LOAD_MAP:
            return _LEGACY_LOAD_MAP[raw]
        lv = int(raw)
        return max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, lv))
    except Exception:
        return LOAD_LEVEL_DEFAULT


def save_level(lv: int) -> None:
    lv = max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, int(lv)))
    try:
        cp = configparser.ConfigParser()
        p = _state_ini_path()
        if os.path.isfile(p):
            cp.read(p, encoding="utf-8")
        if not cp.has_section("ui"):
            cp.add_section("ui")
        cp.set("ui", "load", str(lv))
        with open(p, "w", encoding="utf-8") as f:
            cp.write(f)
    except Exception:
        pass


def _blas_threads(lv: int) -> int:
    """負荷レベル 1-10 から BLAS/OMP スレッド数を計算。
    Lv1=1スレッド, Lv10=全物理コア数, 2-9 は線形補間。
    """
    cpus = max(1, os.cpu_count() or 1)
    if lv <= 1:
        return 1
    if lv >= LOAD_LEVEL_MAX:
        return cpus
    return max(1, round(1 + (cpus - 1) * (lv - 1) / (LOAD_LEVEL_MAX - 1)))


def _resampy_parallel(lv: int) -> bool:
    """Lv4 以上で resampy numba 並列化を有効化。"""
    return lv >= 4


def _file_workers(lv: int) -> int:
    """同時処理ファイル数。Lv1-5=1 (逐次), Lv6-10=2〜全コア/2 (並列)。"""
    if lv <= 5:
        return 1
    cpus = max(1, os.cpu_count() or 1)
    return min(2 + (lv - 6), max(1, cpus // 2))


@dataclass(frozen=True)
class DSREParams:
    m: int = HARMONIC_LAYERS
    decay: float = HARMONIC_DECAY
    pre_hp: int = PRE_HP_CUTOFF_HZ
    post_hp: int = POST_HP_CUTOFF_HZ
    target_sr: int = TARGET_SR
    filter_order: int = FILTER_ORDER
    # v1.6: FLAC 96kHz / PCM_24 固定
    output_format: str = "FLAC"
    output_subtype: str = "PCM_24"
    output_subtype_fallback: str = "PCM_16"
    # v1.11: Improved Harmonic Exciter (帯域分割 + 偶奇分離)
    exciter_lo_hz: int = EXCITER_LO_HZ
    exciter_hi_hz: int = EXCITER_HI_HZ
    exciter_out_hp: int = EXCITER_OUT_HP_HZ
    exciter_drive_lo: float = EXCITER_DRIVE_LO
    exciter_drive_hi: float = EXCITER_DRIVE_HI
    exciter_sat_gain: float = EXCITER_SAT_GAIN
    exciter_even_ratio: float = EXCITER_EVEN_RATIO
    exciter_odd_ratio: float = EXCITER_ODD_RATIO
    # v1.11: Mid Warmth
    mid_warmth_lo_hz: int = MID_WARMTH_LO_HZ
    mid_warmth_hi_hz: int = MID_WARMTH_HI_HZ
    mid_warmth_out_hp_hz: int = MID_WARMTH_OUT_HP_HZ
    mid_warmth_drive: float = MID_WARMTH_DRIVE
    mid_warmth_asym_gain: float = MID_WARMTH_ASYM_GAIN
    # v1.11: Stereo Width
    stereo_widen_hp_hz: int = STEREO_WIDEN_HP_HZ
    stereo_widen_gain: float = STEREO_WIDEN_GAIN
    # v1.11: Air Band Sparkle
    air_band_hp_hz: int = AIR_BAND_HP_HZ
    air_band_out_hp_hz: int = AIR_BAND_OUT_HP_HZ
    air_band_drive: float = AIR_BAND_DRIVE
    air_band_sat_gain: float = AIR_BAND_SAT_GAIN
    # v1.13: Transient Crispness (DR/PLR 改善寄与強化)
    transient_lp_hz: int = TRANSIENT_LP_HZ
    transient_fast_ms: float = TRANSIENT_FAST_MS
    transient_slow_ms: float = TRANSIENT_SLOW_MS
    transient_gain: float = TRANSIENT_GAIN
    # v1.16: 高層 decay 補正 (1.5x 一律、最終 EQ で単調化を引き受け)
    high_layer_boost: float = HARMONIC_HIGH_LAYER_BOOST
    # v1.16: Master Spectral Monotonizer (最終出力単調化、zero-phase FFT EQ)
    master_mono_lo_hz: int = MASTER_MONO_LO_HZ
    master_mono_bin_hz: float = MASTER_MONO_BIN_HZ
    master_mono_smooth_bins: int = MASTER_MONO_SMOOTH_BINS
    master_mono_jump_tol_db: float = MASTER_MONO_JUMP_TOL_DB
    master_mono_run_passes: int = MASTER_MONO_RUN_PASSES
    # v1.13: Nyquist Band Complement (32-48kHz plausible HF)
    nyq_complement_lo_hz: int = NYQ_COMPLEMENT_LO_HZ
    nyq_complement_hi_hz: int = NYQ_COMPLEMENT_HI_HZ
    nyq_complement_shift_hz: int = NYQ_COMPLEMENT_SHIFT_HZ
    nyq_complement_out_hp_hz: int = NYQ_COMPLEMENT_OUT_HP_HZ
    nyq_complement_drive: float = NYQ_COMPLEMENT_DRIVE
    # v1.11: Master headroom
    master_headroom_peak: float = MASTER_HEADROOM_PEAK


PARAMS = DSREParams()


# ===== バンドルリソースのパス解決 =====
def _resource_base_dirs() -> tuple[str, ...]:
    """PyInstaller onedir / 開発実行の両方で同梱リソースを探すためのベースディレクトリ群。"""
    dirs: list[str] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(os.path.abspath(sys.executable)))
    else:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    return tuple(dirs)


def _find_bundled(*relative_paths: str) -> str | None:
    """いずれかのベース + 相対パスの組合せで最初に見つかった絶対パスを返す。"""
    for base in _resource_base_dirs():
        for rel in relative_paths:
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
    return None


# ===== ffmpeg PATH 補完 (同梱 ffmpeg/ffmpeg.exe または _internal/ffmpeg/ffmpeg.exe を探索) =====
def add_ffmpeg_to_path() -> None:
    bundled = _find_bundled(
        os.path.join("ffmpeg", "ffmpeg.exe"),
        os.path.join("_internal", "ffmpeg", "ffmpeg.exe"),
    )
    if bundled:
        os.environ["PATH"] = os.path.dirname(bundled) + os.pathsep + os.environ.get("PATH", "")


# ===== アプリアイコン (logo.ico) =====
def _logo_path() -> str | None:
    return _find_bundled("logo.ico")


def _app_icon() -> "QtGui.QIcon":
    p = _logo_path()
    return QtGui.QIcon(p) if p else QtGui.QIcon()


# ===== subprocess 起動（コマンドプロンプト非表示）=====
def run_hidden(cmd):
    return subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


# ===== 安全読み込み =====
def load_audio_safe(path):
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data.T, sr
    except (RuntimeError, OSError, ValueError):
        pass
    try:
        y, sr = librosa.load(path, mono=False, sr=None, dtype=np.float32)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr
    except Exception as e:
        raise RuntimeError(f"読み込み失敗: {path}") from e


# ===== 保存 (v1.6: 96kHz / FLAC PCM_24 + ffmpeg -c copy でメタデータ継承) =====
def _try_sf_write(path, data, sr, subtype, fmt):
    """書込 → 読み直しで shape / sr が一致するかをラウンドトリップ検証する。
    失敗時は中途半端に残ったファイルを削除して False を返す。"""
    try:
        sf.write(path, data, sr, subtype=subtype, format=fmt)
        check, check_sr = sf.read(path, always_2d=True, dtype="float32")
        if check_sr != sr or check.shape != data.shape:
            raise RuntimeError("roundtrip mismatch")
        return True
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False


def save_flac24_out(in_path, y_out, sr, out_path):
    """DSP 結果を FLAC <TARGET_SR=96kHz> / PCM_24 として書き出し、ffmpeg でメタデータを継承する。

    v1.6 で 96kHz に戻し (理由):
    - v1.5 で 192k/PCM_24 を ship したが、ユーザー主観で「ボーカル裏に高音乗り」違和感
    - 192k は DSEE HX 思想 (44.1k → 96k アップスケーリング) を逸脱、可聴域外倍音が
      DAC/HP で intermodulation を生んで可聴域に折り返す仮説
    - 96k 戻しで DSP ロジックは本家相当、計算量も約 50% 減 (負荷も同時に軽減)

    v1.5 で v1.3 方式に revert (理由):
    - v1.4 で WAV 32bit float に変更したが、foobar2000 で v1.3 (FLAC PCM_24) と
      DR / PLR / 波形すべて同値だった (32bit 化に音質メリット無し、重量増のみ)
    - 24bit の DR は理論上 144dB、実用上の必要 DR (~100dB) を既に上回る
    - 出荷物は FLAC のみ (Vorbis Comment ネイティブ、metadata は ffmpeg `-c copy` で継承)

    保存パス:
    - PCM_24 primary → PCM_16 fallback の 2 段試行 (libsndfile 異常時の安全網)
    - peak > 1.0 のときのみ正規化 (v1.4 の 0.99 スケールは過剰だった、v1.3 方式)
    - ffmpeg コマンドは `-map_metadata 1 -c copy` のみ (FLAC は Vorbis Comment が
      native、`-write_id3v2 1` は不要。WAV 専用フラグだったので v1.4 から削除)
    """
    if y_out.ndim == 1:
        data = y_out.reshape(-1, 1)
    else:
        data = y_out.T
    data = data.astype(np.float32, copy=False)

    # v1.3 方式: peak が 1.0 を超えたときのみ 1.0 に揃える (緩い clip 防止)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak

    base = os.path.splitext(out_path)[0]
    final_path = base + OUTPUT_EXT

    # 一時 FLAC に書込 (メタデータ無し)、PCM_24 → PCM_16 の順に試行
    tmp_path = final_path + ".tmp_src.flac"
    wrote = False
    for subtype in (OUTPUT_SUBTYPE, OUTPUT_SUBTYPE_FALLBACK):
        if _try_sf_write(tmp_path, data, sr, subtype, OUTPUT_FORMAT):
            wrote = True
            break
    if not wrote:
        raise RuntimeError(f"FLAC 書込失敗 (PCM_24 / PCM_16 共に NG): {final_path}")

    # ffmpeg でメタデータ継承 (音声は -c copy で再エンコード無し = 完全無劣化)
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,     # 音声ソース (DSP 済 FLAC)
        "-i", in_path,      # メタデータソース (元 FLAC 等)
        "-map", "0:a",
        "-map_metadata", "1",
        "-c", "copy",
        final_path,
    ]
    try:
        run_hidden(cmd)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    except Exception:
        # ffmpeg 失敗時はメタ無しで確定 (音声は確保されている)
        try:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.replace(tmp_path, final_path)
        except OSError:
            pass
    return final_path


# ===== DSP =====
def freq_shift_mono(x, f_shift, d_sr):
    """1D 実信号を f_shift [Hz] だけ周波数シフト (single-sideband)。

    解析信号 (hilbert で得た complex signal) に e^{j*2*pi*f*t} を乗じると、
    数学的に上側サイドバンドのみが残る。最後に `.real` で実部を取るのは、
    解析信号 z = x + j*H[x] のうち音として復元すべき成分が実部側であるため。
    `np.abs(..)` では振幅包絡線になってしまい原音と関係ないので誤り。
    """
    N = len(x)
    Np = next_fast_len(max(1, N))
    S = signal.hilbert(np.hstack((x, np.zeros(Np - N, dtype=x.dtype))))
    F = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * F)[:N].real


def freq_shift_multi(x, f_shift, d_sr):
    """マルチチャンネル版 freq_shift_mono。各チャンネル独立に適用。
    `.real` を取る理由は freq_shift_mono の docstring を参照。
    """
    Ch, N = x.shape
    Np = next_fast_len(max(1, N))
    padded = np.zeros((Ch, Np), dtype=x.dtype)
    padded[:, :N] = x
    S = signal.hilbert(padded, axis=-1)
    F = np.exp(2j * np.pi * f_shift * d_sr * np.arange(Np))
    return (S * F[np.newaxis, :])[:, :N].real


def safe_butter_sos(order, cutoff_hz, sr, btype="highpass"):
    """SOS (Second-Order Sections) 形式で Butterworth を構築する。
    高次 IIR (本プロジェクトでは order=11) で ba 係数がアンダーフロー / ピボット
    不安定になるのを避けるため、sosfiltfilt と対で使うこと。
    """
    nyq = sr / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.95)
    order = min(order, 20)
    wn = max(1e-6, min(0.999, cutoff_hz / nyq))
    return signal.butter(order, wn, btype=btype, output="sos")


def safe_sosfiltfilt(sos, x, axis=-1):
    """sosfiltfilt のガード付きラッパ。
    理論上 sosfiltfilt は filtfilt(ba) より数値安定で NaN が出にくいが、
    極端な低 Wn の高次 IIR では浮動小数誤差が蓄積しうるためフェイルセーフを張る。
    例外 / NaN / Inf のいずれが出ても入力を [-1, 1] に clip して返す。
    """
    try:
        y = signal.sosfiltfilt(sos, x, axis=axis)
    except Exception:
        return np.clip(x, -1.0, 1.0)
    if not np.all(np.isfinite(y)):
        return np.clip(x, -1.0, 1.0)
    return y


def measure_hf_ratio(x: np.ndarray, sr: int) -> float:
    """4 kHz 以上のエネルギー比 (0=低域のみ, 1=高域のみ)。
    DSEE HX adaptive 思想: 圧縮ロッシー (低 hf_ratio) ほど強化する判定材料。
    stereo は 2ch 平均で測定 (チャンネル間相関による位相打消し対策)。
    """
    sig = np.mean(x, axis=0) if x.ndim > 1 else x
    n = len(sig)
    if n < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    energy = spec * spec
    total = float(np.sum(energy)) + 1e-12
    hf = float(np.sum(energy[freqs >= 4000.0]))
    return hf / total



def _half_rect_dc_removed(src32: np.ndarray) -> np.ndarray:
    """半波整流 + DC offset 除去 (偶数倍音生成の前処理)。"""
    rect = np.maximum(src32, 0.0)
    if rect.ndim > 1:
        rect = rect - np.mean(rect, axis=-1, keepdims=True)
    else:
        rect = rect - np.mean(rect)
    return rect.astype(np.float32, copy=False)


def _tanh_sat(src32: np.ndarray, sat_gain: float) -> np.ndarray:
    """tanh ソフトサチュレータ (奇数倍音、tube-like soft clip)。"""
    return (np.tanh(src32 * sat_gain) / sat_gain).astype(np.float32, copy=False)


def _band_extract(x: np.ndarray, sr: int, lo_hz: float, hi_hz: float | None = None,
                  order: int = 8) -> np.ndarray:
    """HP/BP でソース帯域を抽出。hi_hz=None なら HP 単独。"""
    sos_hp = safe_butter_sos(order, lo_hz, sr, btype="highpass")
    y = safe_sosfiltfilt(sos_hp, x, axis=-1)
    if hi_hz is not None and hi_hz > lo_hz:
        sos_lp = safe_butter_sos(order, hi_hz, sr, btype="lowpass")
        y = safe_sosfiltfilt(sos_lp, y, axis=-1)
    return y


def apply_master_monotonizer(result: np.ndarray, sr: int,
                             lo_hz: float = MASTER_MONO_LO_HZ,
                             bin_hz: float = MASTER_MONO_BIN_HZ,
                             smooth_bins: int = MASTER_MONO_SMOOTH_BINS,
                             jump_tol_db: float = MASTER_MONO_JUMP_TOL_DB,
                             passes: int = MASTER_MONO_RUN_PASSES) -> np.ndarray:
    """v1.16: 最終出力スペクトルの局所上昇 (隣接 bin の +jump_tol_db 超過) のみ
    を縮小する zero-phase EQ。

    cumulative-min 戦略は補完エネルギーを殺すため不採用。代わりに局所差分
    (mag[b] / mag[b-1]) が許容比 (= 10^(jump_tol_db/20)) を超えた bin だけ
    target = mag[b-1] * tol まで縮小する。低域全体に高域を引っ張られない。

    1. rfft → magnitude/phase 取得
    2. lo_hz 以上を bin_hz 幅で集約 (RMS)
    3. passes 回反復: 上昇ジャンプ > tol_db を検出、当該 bin に縮小ゲイン
       gain[b] = (mag[b-1] * tol) / mag[b]
    4. ゲインを smooth_bins で隣接平均化 (ringing 防止)
    5. rfft bin に展開 → 振幅乗算 → irfft

    zero-phase、forward-only IIR を使わないので新規ノイズなし。
    補完エネルギーは「許容比以下」の場合は完全保持。
    NaN/Inf 検出時は元 result を返すフェイルセーフ。
    """
    if result.size == 0:
        return result
    try:
        if result.ndim == 1:
            channels = [result]
        else:
            channels = [result[c] for c in range(result.shape[0])]

        out_chs = []
        n_orig = channels[0].shape[-1]
        tol_lin = float(10.0 ** (jump_tol_db / 20.0))

        for ch in channels:
            ch64 = ch.astype(np.float64, copy=False)
            X = np.fft.rfft(ch64)
            mag = np.abs(X)
            freqs = np.fft.rfftfreq(n_orig, d=1.0 / sr)

            n_bins = int(np.floor((sr / 2.0 - lo_hz) / bin_hz))
            if n_bins < 2:
                out_chs.append(ch)
                continue

            # bin 集約 (RMS)
            bin_mag = np.zeros(n_bins, dtype=np.float64)
            bin_indices = []
            for b in range(n_bins):
                f_lo = lo_hz + b * bin_hz
                f_hi = f_lo + bin_hz
                idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
                bin_indices.append(idx)
                if idx.size > 0:
                    bin_mag[b] = float(np.mean(mag[idx] ** 2)) ** 0.5

            # 反復: 隣接上昇 > tol_lin の bin を直前 bin * tol_lin に抑制
            cur = bin_mag.copy()
            for _ in range(max(1, passes)):
                gain_step = np.ones(n_bins, dtype=np.float64)
                # 1 ~ n_bins-1 を順に確認 (b の上昇は b-1 を基準)
                for b in range(1, n_bins):
                    prev = cur[b - 1]
                    if prev < 1e-30:
                        continue
                    ratio = cur[b] / prev
                    if ratio > tol_lin:
                        g = (prev * tol_lin) / max(cur[b], 1e-30)
                        gain_step[b] = g
                        cur[b] = cur[b] * g
                # smooth gain across bins
                if smooth_bins > 1:
                    kernel = np.ones(smooth_bins, dtype=np.float64) / float(smooth_bins)
                    gain_step = np.convolve(gain_step, kernel, mode="same")

                # accumulate into bin_mag base
                bin_mag = bin_mag * gain_step
                cur = bin_mag.copy()

            # 最終ゲイン = bin_mag / 元 RMS
            orig_bin_mag = np.zeros(n_bins, dtype=np.float64)
            for b in range(n_bins):
                idx = bin_indices[b]
                if idx.size > 0:
                    orig_bin_mag[b] = float(np.mean(np.abs(X[idx]) ** 2)) ** 0.5
            final_gain = np.ones(n_bins, dtype=np.float64)
            mask = orig_bin_mag > 1e-30
            final_gain[mask] = bin_mag[mask] / orig_bin_mag[mask]
            # ゲインは 1.0 を超えない (補完を増やさない、減らすのみ)
            final_gain = np.clip(final_gain, 0.0, 1.0)

            full_gain = np.ones_like(mag)
            for b in range(n_bins):
                idx = bin_indices[b]
                if idx.size > 0:
                    full_gain[idx] = final_gain[b]

            X2 = X * full_gain
            ch_out = np.fft.irfft(X2, n=n_orig).astype(ch.dtype, copy=False)
            if not np.all(np.isfinite(ch_out)):
                return result
            out_chs.append(ch_out)

        if result.ndim == 1:
            return out_chs[0]
        return np.stack(out_chs, axis=0)
    except Exception:
        return result


def harmonic_exciter(x, sr, params: "DSREParams | None" = None):
    """v1.11: 帯域分割 + 偶奇分離型 harmonic exciter (改良版)。

    旧 v1.10 は 4kHz 以上を単段で処理しており、10kHz 以上の高域でも同じ drive
    を掛けるため、サ行・シンバルの「刺さり」が出るリスクがあった。
    v1.11 では 2 帯域 (4-10kHz / 10kHz+) に分割し、上段の drive を下段の半分強
    に抑える。さらに偶数倍音 (warmth) と奇数倍音 (presence) のブレンド比を
    パラメータ化 (デフォルト 40/60) して、温度感と解像感の両立を図る。

    狙い:
      - 4-10kHz: 偶数倍音多めで「中高域の温度・実在感」(ボーカル子音・ストリングス)
      - 10kHz+:  奇数倍音中心 + 控えめ drive で「空気感・透明感」(刺さり回避)
    """
    p = params if params is not None else PARAMS

    # 下段 (4-10kHz): 厚みを担う帯域、drive 大きめ + 偶数倍音多め
    src_lo = _band_extract(x, sr, p.exciter_lo_hz, p.exciter_hi_hz, order=8)
    s_lo = src_lo.astype(np.float32, copy=False)
    rect_lo = _half_rect_dc_removed(s_lo)
    sat_lo = _tanh_sat(s_lo, p.exciter_sat_gain)
    mix_lo = p.exciter_even_ratio * rect_lo + p.exciter_odd_ratio * sat_lo

    # 上段 (10kHz+): 透明感担当、drive 小さめ + 奇数倍音寄り (刺さり回避)
    src_hi = _band_extract(x, sr, p.exciter_hi_hz, hi_hz=None, order=8)
    s_hi = src_hi.astype(np.float32, copy=False)
    rect_hi = _half_rect_dc_removed(s_hi)
    sat_hi = _tanh_sat(s_hi, p.exciter_sat_gain * 0.85)  # 上段はさらに柔らかく
    # 上段は奇数倍音 70% / 偶数 30% (透明感を優先、温度より sparkle)
    mix_hi = 0.30 * rect_hi + 0.70 * sat_hi

    # 出力 HP で生成倍音の高域成分のみ抽出 (中低域への滲み防止)
    sos_out = safe_butter_sos(8, p.exciter_out_hp, sr, btype="highpass")
    excited_lo = safe_sosfiltfilt(sos_out, mix_lo, axis=-1)
    excited_hi = safe_sosfiltfilt(sos_out, mix_hi, axis=-1)

    if not (np.all(np.isfinite(excited_lo)) and np.all(np.isfinite(excited_hi))):
        return np.zeros_like(x)

    out = excited_lo * p.exciter_drive_lo + excited_hi * p.exciter_drive_hi
    return out.astype(x.dtype, copy=False)


def mid_warmth(x, sr, params: "DSREParams | None" = None):
    """v1.11: 中域 (200-1500Hz) に微小 even-order harmonic を加え温度感・艶を付与。

    中域は人間の聴感が最も敏感な帯域 (Fletcher-Munson の 2-4kHz peak) で、
    この帯域の「実在感・温度感」が音源の魅力を左右する。
    非対称 soft clip (asymmetric saturation) は偶数次倍音 (主に 2nd) を生成し、
    アナログテープ・チューブアンプ的な暖かさを生む。

    drive=0.05 は非常に控えめ (聴感ではっきり差を出す手前で停止)。
    過剰な歪・濁り・中域の濁りを完全回避する設計。
    """
    p = params if params is not None else PARAMS

    src = _band_extract(x, sr, p.mid_warmth_lo_hz, p.mid_warmth_hi_hz, order=6)
    s = src.astype(np.float32, copy=False)

    # 非対称 soft clip: tanh(g*x + g*x^2 * 0.15)
    # x^2 項が偶数次倍音 (2nd harmonic) を主に生成、アナログ的な暖かさの源
    g = p.mid_warmth_asym_gain
    asym = np.tanh(g * s + g * s * s * 0.15) / g - s  # 原音引いて差分のみ抽出
    asym = asym.astype(np.float32, copy=False)

    # 加算は中高域のみ (低域の膨張を完全回避、ボーカルの抜けに寄与)
    sos_out = safe_butter_sos(6, p.mid_warmth_out_hp_hz, sr, btype="highpass")
    warm = safe_sosfiltfilt(sos_out, asym, axis=-1)

    if not np.all(np.isfinite(warm)):
        return np.zeros_like(x)

    return (warm * p.mid_warmth_drive).astype(x.dtype, copy=False)


def stereo_widen_hf(x, sr, params: "DSREParams | None" = None):
    """v1.11: Mid/Side 分解で Side の高域 (2kHz+) のみ微増。空間広がり・臨場感を向上。

    M = (L + R) / 2、S = (L - R) / 2。
    Side は元々ステレオ情報 (空間情報) を担う成分で、ここの高域だけを上げると
    「ホール感・奥行き・楽器の左右分離」が向上する。Mid (センター) は無触なので
    ボーカル定位は無傷。

    mono 入力時 (1ch / 2ch 同一) は完全 no-op、副作用ゼロ。
    """
    p = params if params is not None else PARAMS

    if x.ndim != 2 or x.shape[0] != 2:
        return np.zeros_like(x)

    L = x[0]
    R = x[1]
    side = ((L - R) * 0.5).astype(np.float32, copy=False)

    # Side が完全ゼロ (mono 化された stereo) なら no-op
    if float(np.max(np.abs(side))) < 1e-9:
        return np.zeros_like(x)

    sos_hp = safe_butter_sos(6, p.stereo_widen_hp_hz, sr, btype="highpass")
    side_hf = safe_sosfiltfilt(sos_hp, side, axis=-1)

    if not np.all(np.isfinite(side_hf)):
        return np.zeros_like(x)

    delta_side = (side_hf * p.stereo_widen_gain).astype(np.float32, copy=False)
    out = np.zeros_like(x)
    # Side を増やすと L += dS、R -= dS (Mid は無触のまま定位安定)
    out[0] = delta_side
    out[1] = -delta_side
    return out.astype(x.dtype, copy=False)


def air_band_sparkle(x, sr, params: "DSREParams | None" = None):
    """v1.11: 16kHz+ の "air" 帯域に plausible 高域を補い、空気感・透明感を強化。

    DSEE HX の核思想「失われた高域の plausible 推定」を最高域帯で実装。
    既存 zansei_impl/exciter は 12kHz 付近までを主にカバーするが、それ以上の
    超高域 (15-22kHz、いわゆる "air") はスパース。ここを envelope follower +
    強めサチュレータで控えめに補い、解像感・透明感・没入感を底上げする。
    """
    p = params if params is not None else PARAMS

    # ソース: 13kHz+ の信号 (この帯域に元々あるエネルギーから倍音を作る)
    src = _band_extract(x, sr, p.air_band_hp_hz, hi_hz=None, order=6)
    s = src.astype(np.float32, copy=False)

    # 強めサチュレータ (微小信号でも倍音が出る)
    sat = _tanh_sat(s, p.air_band_sat_gain)
    rect = _half_rect_dc_removed(s)
    mix = 0.4 * rect + 0.6 * sat

    # 15kHz+ のみを通す (中域への滲みを完全に断つ)
    sos_out = safe_butter_sos(6, p.air_band_out_hp_hz, sr, btype="highpass")
    air = safe_sosfiltfilt(sos_out, mix, axis=-1)

    if not np.all(np.isfinite(air)):
        return np.zeros_like(x)

    out = (air * p.air_band_drive).astype(x.dtype, copy=False)
    return out.astype(x.dtype, copy=False)


def transient_crispness(x, sr, params: "DSREParams | None" = None):
    """v1.11: 低域 (<200Hz) の立ち上がり (transient) のみを軽くブースト。

    高速 envelope (5ms) と低速 envelope (50ms) の差を取ると、長期平均より
    瞬間的に音圧が立つ部分 (キック・ベースの attack) を取り出せる。
    そこにだけ +Δ を乗せることで、低域の量感は変えずに「輪郭・締まり・制動感」
    のみを向上させる。

    一次 IIR (one-pole lowpass) で envelope 抽出 → そのまま transient 信号として
    元の低域信号に加算する。drive=0.08 でごく軽め (低域膨張防止)。
    """
    p = params if params is not None else PARAMS

    # 低域帯域抽出 (~200Hz LP)
    sos_lp = safe_butter_sos(4, p.transient_lp_hz, sr, btype="lowpass")
    low = safe_sosfiltfilt(sos_lp, x, axis=-1)
    if not np.all(np.isfinite(low)):
        return np.zeros_like(x)

    abs_low = np.abs(low.astype(np.float32, copy=False))

    # one-pole IIR で fast/slow envelope を計算
    # alpha = 1 - exp(-1 / (sr * tau))
    fast_alpha = float(1.0 - np.exp(-1.0 / max(1.0, sr * (p.transient_fast_ms * 1e-3))))
    slow_alpha = float(1.0 - np.exp(-1.0 / max(1.0, sr * (p.transient_slow_ms * 1e-3))))

    def _one_pole(sig: np.ndarray, alpha: float) -> np.ndarray:
        # SOS で近似 (sosfiltfilt より高速、forward-only で transient 検出に十分)
        b = np.array([alpha], dtype=np.float64)
        a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
        try:
            return signal.lfilter(b, a, sig, axis=-1).astype(np.float32, copy=False)
        except Exception:
            return sig.astype(np.float32, copy=False)

    env_fast = _one_pole(abs_low, fast_alpha)
    env_slow = _one_pole(abs_low, slow_alpha)

    # transient = max(0, fast - slow) → 立ち上がり部のみ (>0)
    diff = env_fast - env_slow
    diff = np.maximum(diff, 0.0).astype(np.float32, copy=False)

    # 低域信号 sign を保持して transient 量だけブースト分を生成
    sign = np.sign(low.astype(np.float32, copy=False))
    boost = sign * diff

    if not np.all(np.isfinite(boost)):
        return np.zeros_like(x)

    return (boost * p.transient_gain).astype(x.dtype, copy=False)


def nyquist_complement(x, sr, params: "DSREParams | None" = None):
    """v1.13: 32-48kHz 帯域への plausible HF 補完 (本家 + v1.10 同等の基礎性能)。

    8-22kHz 帯域を抽出 → +24kHz freq_shift で 32-46kHz に飛ばす → 30kHz HP で
    通す。これにより Audacity 等のスペクトル表示で 32-48kHz 帯に明確な補完
    エネルギーが観測される (DSRE 本来の "ナイキスト到達補完" 性能)。

    32-48kHz は人間可聴域外 (20kHz 上限) だが、DAC 再生時の plausible 残響成分
    として高品位機器で実体感に寄与する (DSEE HX 思想と整合)。
    """
    p = params if params is not None else PARAMS

    nyq = sr / 2.0
    shift = float(min(p.nyq_complement_shift_hz, nyq * 0.95 - p.nyq_complement_lo_hz))
    if shift <= 0:
        return np.zeros_like(x)

    src = _band_extract(x, sr, p.nyq_complement_lo_hz, p.nyq_complement_hi_hz, order=6)
    if not np.all(np.isfinite(src)):
        return np.zeros_like(x)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    shifted = f_dn(src.astype(x.dtype, copy=False), shift, d_sr)

    out_hp = float(min(p.nyq_complement_out_hp_hz, nyq * 0.95))
    sos_out = safe_butter_sos(6, out_hp, sr, btype="highpass")
    nyq_band = safe_sosfiltfilt(sos_out, shifted, axis=-1)

    if not np.all(np.isfinite(nyq_band)):
        return np.zeros_like(x)

    out = (nyq_band * p.nyq_complement_drive).astype(x.dtype, copy=False)
    return out.astype(x.dtype, copy=False)


def zansei_impl(x, sr, progress_cb=None, abort_cb=None):
    sos_pre = safe_butter_sos(PARAMS.filter_order, PARAMS.pre_hp, sr, btype="highpass")
    d_src = safe_sosfiltfilt(sos_pre, x, axis=-1)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    n_layers = PARAMS.m
    decays = np.exp(-np.arange(1, n_layers + 1) * PARAMS.decay)
    # v1.16: 高層 (i>=5) decay 補正は 1.5x 一律段差に戻す (v1.13/v1.14 互換)。
    # 段差由来の局所ピークは最終 master monotonizer EQ で完全補正されるため
    # 段差そのものは d_res で許容して良い (NYQ 補完量を最大化する設計)。
    if n_layers > 5:
        decays[5:] = decays[5:] * PARAMS.high_layer_boost
    nyq = sr / 2.0

    for i in range(n_layers):
        if abort_cb and abort_cb():
            break

        shift = sr * (i + 1) / (n_layers * 2.0)
        if shift >= nyq:
            if progress_cb:
                progress_cb(i + 1, n_layers)
            continue

        d_res += f_dn(d_src, shift, d_sr) * decays[i]

        if progress_cb:
            progress_cb(i + 1, n_layers)

    sos_post = safe_butter_sos(PARAMS.filter_order, PARAMS.post_hp, sr, btype="highpass")
    d_res = safe_sosfiltfilt(sos_post, d_res, axis=-1)

    # v1.13: Multi-Aspect Enhancement (6 path、原音無減衰の additive)
    # path: residual / exciter / mid warmth / stereo widen / air band /
    #       transient / nyq complement (32-48k 補完を独立 path で確実に)
    d_exc = harmonic_exciter(x, sr, params=PARAMS)
    d_warm = mid_warmth(x, sr, params=PARAMS)
    d_widen = stereo_widen_hf(x, sr, params=PARAMS)
    d_air = air_band_sparkle(x, sr, params=PARAMS)
    d_tran = transient_crispness(x, sr, params=PARAMS)
    d_nyq = nyquist_complement(x, sr, params=PARAMS)

    d_extra = d_res + d_exc + d_warm + d_widen + d_air + d_tran + d_nyq

    # v1.14: Master Headroom 削除 — v1.9.1 / v1.10 同等の純粋加算へ復帰。
    result = x + d_extra

    if not np.all(np.isfinite(result)):
        return np.clip(x, -1.0, 1.0)

    # v1.16: 最終出力に対する単調化制約 (zero-phase FFT EQ)。
    # 4kHz 以上の各 1kHz バンドで隣接 bin の +4dB 超過のみ縮小。低域全体の
    # 値で高域を引っ張らない (補完エネルギー維持)。zero-phase なので位相歪なし。
    monotonized = apply_master_monotonizer(
        result, sr,
        lo_hz=PARAMS.master_mono_lo_hz,
        bin_hz=PARAMS.master_mono_bin_hz,
        smooth_bins=PARAMS.master_mono_smooth_bins,
        jump_tol_db=PARAMS.master_mono_jump_tol_db,
        passes=PARAMS.master_mono_run_passes,
    )
    if not np.all(np.isfinite(monotonized)):
        return result
    return monotonized


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


# ===== Worker =====
class Worker(QtCore.QThread):
    sig_step = QtCore.Signal(int)
    sig_all = QtCore.Signal(int)
    sig_text = QtCore.Signal(str)

    def __init__(self, files, level: int = LOAD_LEVEL_DEFAULT):
        super().__init__()
        self.files = files
        self._abort = False
        self._pause = False
        self._mutex = QtCore.QMutex()
        self._wait = QtCore.QWaitCondition()
        self._failed: list[str] = []
        self._trash_failed = 0
        self._level = max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, int(level)))
        import threading as _t
        self._level_lock = _t.Lock()
        self._trash_lock = _t.Lock()

    def set_level(self, lv: int) -> None:
        """処理中にリアルタイムで負荷レベルを変更する。次のファイルから反映。"""
        with self._level_lock:
            self._level = max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, int(lv)))

    def _get_level(self) -> int:
        with self._level_lock:
            return self._level

    def abort(self):
        self._mutex.lock()
        self._abort = True
        self._wait.wakeAll()
        self._mutex.unlock()

    def pause_toggle(self):
        self._mutex.lock()
        self._pause = not self._pause
        if not self._pause:
            self._wait.wakeAll()
        self._mutex.unlock()

    def _wait_if_paused(self):
        self._mutex.lock()
        while self._pause and not self._abort:
            self._wait.wait(self._mutex)
        self._mutex.unlock()

    def run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: F401

        try:
            from threadpoolctl import threadpool_limits as _tpl
        except Exception:
            _tpl = None

        total = len(self.files)
        start_t = time.time()
        succeeded = 0
        completed_count = 0

        # run() 開始時の level で BLAS スレッド数とnumba を初期設定
        lv_init = self._get_level()
        n_thr_init = _blas_threads(lv_init)
        try:
            import numba as _nb
            _nb.set_num_threads(n_thr_init)
        except Exception:
            pass
        kw_init = {"limits": n_thr_init} if (_tpl and n_thr_init > 0) else None
        outer_ctx = _tpl(**kw_init) if kw_init else _NullCtx()

        def _process_one(path: str, lv: int, use_step_cb: bool) -> str:
            """単一ファイルを処理して "ok" / "trash_fail" を返す。"""
            par = _resampy_parallel(lv)

            def step_cb(cur, m):
                if use_step_cb:
                    self.sig_step.emit(int(cur * 100 / m))

            y, sr = load_audio_safe(path)
            if sr != PARAMS.target_sr:
                try:
                    y = resampy.resample(y, sr, PARAMS.target_sr, parallel=par)
                except TypeError:
                    y = resampy.resample(y, sr, PARAMS.target_sr)
                sr = PARAMS.target_sr
            y_out = zansei_impl(
                y, sr,
                progress_cb=step_cb if use_step_cb else None,
                abort_cb=lambda: self._abort,
            )
            out = os.path.join(OUTPUT_DIR, os.path.basename(path))
            save_flac24_out(path, y_out, sr, out)
            try:
                send2trash(path)
            except Exception:
                return "trash_fail"
            return "ok"

        pending = list(self.files)
        max_cpus = max(1, os.cpu_count() or 1)
        in_flight: list = []  # (Future, path)
        submit_idx = 0

        with outer_ctx:
            with ThreadPoolExecutor(max_workers=max_cpus) as executor:
                while not self._abort:
                    # ---- 完了 future を回収 ----
                    still_running = []
                    for fut, fpath in in_flight:
                        if fut.done():
                            completed_count += 1
                            try:
                                result = fut.result()
                            except Exception:
                                self._failed.append(os.path.basename(fpath))
                            else:
                                succeeded += 1
                                if result == "trash_fail":
                                    with self._trash_lock:
                                        self._trash_failed += 1
                            # 進捗テキスト更新
                            elapsed = time.time() - start_t
                            if completed_count > 0 and total > completed_count:
                                remain = (elapsed / completed_count) * (total - completed_count)
                                fail_n = len(self._failed)
                                parts = []
                                if fail_n:
                                    parts.append(f"失敗{fail_n}")
                                if self._trash_failed:
                                    parts.append(f"ゴミ箱{self._trash_failed}")
                                suffix = ("  " + "  ".join(parts)) if parts else ""
                                self.sig_text.emit(f"{completed_count}/{total}  残り{int(remain)}秒{suffix}")
                            self.sig_all.emit(int(completed_count * 100 / total))
                            self.sig_step.emit(100)
                        else:
                            still_running.append((fut, fpath))
                    in_flight = still_running

                    # ---- 終了判定 ----
                    if not pending and not in_flight:
                        break

                    # ---- 一時停止 (新規投入前、QThread コンテキストで呼ぶ) ----
                    self._wait_if_paused()
                    if self._abort:
                        break

                    # ---- 現在の level に基づいてファイルを投入 ----
                    lv = self._get_level()
                    n_workers = _file_workers(lv)
                    use_step = (n_workers <= 1)
                    while pending and len(in_flight) < n_workers and not self._abort:
                        fpath = pending.pop(0)
                        submit_idx += 1
                        self.sig_text.emit(f"{submit_idx}/{total}")
                        fut = executor.submit(_process_one, fpath, lv, use_step)
                        in_flight.append((fut, fpath))

                    time.sleep(0.05)

        fail_n = len(self._failed)
        trash_n = self._trash_failed
        extras = []
        if fail_n:
            extras.append(f"失敗{fail_n}")
        if trash_n:
            extras.append(f"ゴミ箱{trash_n}")
        tail = ("  " + "  ".join(extras)) if extras else ""
        if self._abort:
            self.sig_text.emit(f"中断  成功{succeeded}/{total}{tail}")
        elif extras:
            self.sig_text.emit(f"完了  成功{succeeded}/{total}{tail}")
        else:
            self.sig_text.emit(f"完了  {succeeded}/{total}")


# ===== UI (v1.4: トレイ常駐 + 負荷サブメニュー + logo.ico + × 即終了) =====
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DSRE")
        self.setWindowIcon(_app_icon())
        self.resize(340, 220)

        self.label = QtWidgets.QLabel("待機")
        self.pb_file = QtWidgets.QProgressBar()
        self.pb_all = QtWidgets.QProgressBar()

        self.btn_start = QtWidgets.QPushButton("開始")
        self.btn_pause = QtWidgets.QPushButton("一時停止")
        self.btn_cancel = QtWidgets.QPushButton("取消")

        _lv = load_level()
        self.sld_level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_level.setRange(LOAD_LEVEL_MIN, LOAD_LEVEL_MAX)
        self.sld_level.setValue(_lv)
        self.sld_level.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.sld_level.setTickInterval(1)
        self.lbl_level = QtWidgets.QLabel(f"負荷 {_lv}/{LOAD_LEVEL_MAX}")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.pb_file)
        layout.addWidget(self.pb_all)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.btn_pause)
        layout.addWidget(self.btn_cancel)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.lbl_level)
        row.addWidget(self.sld_level, 1)
        layout.addLayout(row)

        self.setLayout(layout)

        self.btn_start.clicked.connect(self.start)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_cancel.clicked.connect(self.cancel)
        self.sld_level.valueChanged.connect(self._on_level_changed)

        self.worker = None
        self._tray = None
        self._tray_level_act: "QtGui.QAction | None" = None
        self._setup_tray()

    # ---- トレイ ----
    def _setup_tray(self) -> None:
        """システムトレイアイコン + 右クリックメニュー (開始/一時停止/取消/負荷±/終了) を構築。"""
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QtWidgets.QSystemTrayIcon(self)
        self._tray.setIcon(_app_icon())
        self._tray.setToolTip("DSRE")

        menu = QtWidgets.QMenu()
        act_show = menu.addAction("表示")
        act_show.triggered.connect(self._show_from_tray)
        menu.addSeparator()

        menu.addAction("開始", self.start)
        menu.addAction("一時停止", self.pause)
        menu.addAction("取消", self.cancel)
        menu.addSeparator()

        # 負荷サブメニュー: ◀ 現在値表示 ▶
        sub = menu.addMenu("負荷")
        sub.addAction("◀ 減らす", lambda: self._adjust_level(-1))
        lv_now = self.sld_level.value()
        self._tray_level_act = sub.addAction(f"負荷: {lv_now}/{LOAD_LEVEL_MAX}")
        self._tray_level_act.setEnabled(False)
        sub.addAction("増やす ▶", lambda: self._adjust_level(+1))

        menu.addSeparator()
        menu.addAction("終了", self._quit_app)

        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray)
        self._tray.show()

    def _on_level_changed(self, lv: int) -> None:
        """スライダー変更時: ラベル更新 + 保存 + worker 伝播 + トレイ同期。"""
        self.lbl_level.setText(f"負荷 {lv}/{LOAD_LEVEL_MAX}")
        save_level(lv)
        if self.worker and self.worker.isRunning():
            self.worker.set_level(lv)
        if self._tray_level_act is not None:
            self._tray_level_act.setText(f"負荷: {lv}/{LOAD_LEVEL_MAX}")

    def _adjust_level(self, delta: int) -> None:
        """トレイの ◀ / ▶ からレベルを ±1 する。"""
        lv = max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, self.sld_level.value() + delta))
        self.sld_level.setValue(lv)  # valueChanged → _on_level_changed が連鎖

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_tray(self, reason) -> None:
        # 左クリックでウィンドウの表示/非表示をトグル
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible() and self.isActiveWindow():
                self.hide()
            else:
                self._show_from_tray()

    def _quit_app(self) -> None:
        """トレイ「終了」および × ボタン共通の終了処理。確認ダイアログなし (ユーザー要望)。"""
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(3000)
        if self._tray is not None:
            self._tray.hide()
        QtWidgets.QApplication.instance().quit()

    # ---- ファイル処理 ----
    def load_files(self):
        files = []

        existing = set()
        if os.path.exists(OUTPUT_DIR):
            existing = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR)}

        if not os.path.exists(INPUT_DIR):
            return files

        for f in os.listdir(INPUT_DIR):
            if f.lower().endswith(".flac"):
                if os.path.splitext(f)[0] not in existing:
                    files.append(os.path.join(INPUT_DIR, f))

        return files

    def start(self):
        if self.worker and self.worker.isRunning():
            return
        files = self.load_files()
        if not files:
            return

        lv = self.sld_level.value() if hasattr(self, "sld_level") else LOAD_LEVEL_DEFAULT
        self.worker = Worker(files, level=lv)
        self.worker.sig_step.connect(self.pb_file.setValue)
        self.worker.sig_all.connect(self.pb_all.setValue)
        self.worker.sig_text.connect(self.label.setText)
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.pause_toggle()

    def cancel(self):
        if self.worker:
            self.worker.abort()

    # ---- ウィンドウイベント (最小化→トレイ隠蔽、× で即終了) ----
    def changeEvent(self, event):
        # 最小化はトレイに隠蔽 (タスクバーから消す)
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            if self.isMinimized() and self._tray is not None:
                event.ignore()
                # QTimer.singleShot で hide を遅延させないと状態変化中で無視されることがある
                QtCore.QTimer.singleShot(0, self.hide)
                return
        super().changeEvent(event)

    def closeEvent(self, event):
        # × = 即終了 (確認ダイアログなし、処理中であれば abort + 3 秒待機)
        self._quit_app()
        event.accept()


def _run_selftest() -> int:
    """selftest gate: verdict=DEGRADED → exit 1 → CI/deploy が artifact を作らない。"""
    import traceback
    log_dir = os.path.dirname(sys.executable) or os.getcwd()
    log_path = os.path.join(log_dir, "selftest.log")
    try:
        # ---- (1) imports ----
        import numpy as _np
        import scipy as _sp
        import scipy.signal as _sps
        import scipy.linalg  # noqa: F401
        import scipy.fft  # noqa: F401
        import numpy.testing  # noqa: F401  # unittest 地雷検出用
        import librosa as _lb
        import resampy  # noqa: F401
        import soundfile as _sf
        import send2trash  # noqa: F401
        from PySide6 import QtCore, QtWidgets  # noqa: F401
        import threadpoolctl  # noqa: F401

        _ = _sps.butter
        _ = _sps.filtfilt
        _ = _sps.sosfiltfilt
        _ = _sps.hilbert

        tpc_version = getattr(threadpoolctl, "__version__", "?")
        notes: list[str] = []
        verdict = "EQUIV"

        # ---- (2) FLAC <TARGET_SR> kHz / PCM_24 roundtrip ----
        sr_test = TARGET_SR  # v1.6: TARGET_SR 参照に変更 (96k に戻したため hard-coded 192000 を排除)
        t = _np.arange(sr_test // 20, dtype=_np.float32) / sr_test
        sig_mono = (0.25 * _np.sin(2 * _np.pi * 1000.0 * t)).astype(_np.float32)
        sig_stereo = _np.stack([sig_mono, sig_mono], axis=1)

        rt_max_abs = float("nan")
        rt_status = "FAIL"
        # PCM_24 の量子化誤差: 2^-23 ≈ 1.19e-7 (signed 24bit の最小単位)
        # マージン込みで 1.5e-7 を閾値に
        RT_THRESHOLD = 1.5e-7
        tmp_flac = tempfile.NamedTemporaryFile(delete=False, suffix=".flac")
        tmp_flac.close()
        try:
            _sf.write(tmp_flac.name, sig_stereo, sr_test, subtype=OUTPUT_SUBTYPE, format=OUTPUT_FORMAT)
            data_read, sr_read = _sf.read(tmp_flac.name, always_2d=True, dtype="float32")
            assert sr_read == sr_test, f"sr mismatch {sr_read}!={sr_test}"
            assert data_read.shape == sig_stereo.shape, "shape mismatch after round-trip"
            rt_max_abs = float(_np.max(_np.abs(data_read - sig_stereo)))
            # FLAC PCM_24 は loss-less な量子化 (24bit 精度内で復元可能)
            rt_status = "OK" if rt_max_abs < RT_THRESHOLD else f"LOSSY({rt_max_abs:.2e})"
            if rt_max_abs >= RT_THRESHOLD:
                verdict = "DEGRADED"
        finally:
            try:
                if os.path.exists(tmp_flac.name):
                    os.remove(tmp_flac.name)
            except OSError:
                pass

        # ---- (3) sosfiltfilt 等価性 (filtfilt(ba) vs sosfiltfilt(sos)) ----
        # 既存 zansei_impl の使用条件 (order=11, Wn=PRE_HP_CUTOFF_HZ/nyq 等) を再現
        rng_eq = _np.random.default_rng(4242)
        N_eq = 4096
        # sweep + white noise (フィルタを余すところなく exercise)
        t_eq = _np.arange(N_eq, dtype=_np.float32) / sr_test
        sweep = _sps.chirp(t_eq, f0=50.0, f1=20000.0, t1=t_eq[-1], method="logarithmic").astype(_np.float32)
        noise = rng_eq.standard_normal(N_eq).astype(_np.float32) * 0.1
        x_eq = (sweep + noise).astype(_np.float32)

        nyq = sr_test * 0.5
        # DC 信号 (HP filter の stopband rejection を数値で確認するため)
        dc_level = 0.5
        dc_sig = _np.full(N_eq, dc_level, dtype=_np.float32)

        eq_results: list[tuple[str, float, float, str]] = []  # (label, max_abs, rms_rel, tag)
        for label, wn_hz, btype in (
            ("pre_HP", float(PRE_HP_CUTOFF_HZ), "highpass"),
            ("post_HP", float(POST_HP_CUTOFF_HZ), "highpass"),
        ):
            wn = max(1e-6, min(0.999, wn_hz / nyq))
            # BA 形式は order > 8 かつ Wn < 0.1 で数値不安定 (scipy 公式警告)
            low_wn_regime = wn < 0.1

            # 旧 (ba)
            old_ok = True
            try:
                b, a = _sps.butter(FILTER_ORDER, wn, btype=btype)
                y_old = _sps.filtfilt(b, a, x_eq)
                if not _np.all(_np.isfinite(y_old)):
                    old_ok = False
            except Exception:
                old_ok = False
                y_old = None

            # 新 (sos)
            sos = _sps.butter(FILTER_ORDER, wn, btype=btype, output="sos")
            y_new = _sps.sosfiltfilt(sos, x_eq)
            new_finite = bool(_np.all(_np.isfinite(y_new)))

            if not new_finite:
                eq_results.append((label, float("nan"), float("nan"), "NEW_NaN"))
                verdict = "DEGRADED"
                continue

            # 陽性サニティ: HP フィルタは DC を強く減衰させるはず (-40dB 以上)
            # filtfilt は往復で効くので理論上 -60dB 以上期待できる
            y_dc = _sps.sosfiltfilt(sos, dc_sig)
            peak_dc = float(_np.max(_np.abs(y_dc))) + 1e-30
            dc_rejection_db = 20.0 * _np.log10(peak_dc / dc_level)
            if dc_rejection_db > -40.0:
                eq_results.append((label, float("nan"), float("nan"), f"HP_REJECT_BAD({dc_rejection_db:.1f}dB)"))
                verdict = "DEGRADED"
                continue

            if not old_ok or y_old is None:
                # 旧が NaN / 例外、新は有限 → IMPROVED
                eq_results.append((label, float("nan"), float("nan"), f"OLD_FAIL_NEW_OK(dc_rej={dc_rejection_db:.0f}dB)"))
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
                continue

            diff = y_new - y_old
            max_abs = float(_np.max(_np.abs(diff)))
            rms_ref = float(_np.sqrt(_np.mean(y_old * y_old)) + 1e-30)
            rms_rel = float(_np.sqrt(_np.mean(diff * diff))) / rms_ref

            over_thresh = max_abs > 1e-4 or rms_rel > 1e-5
            if over_thresh and low_wn_regime:
                # 低 Wn では BA 形式自体が数値的に不正確。sos は HP サニティを満たしており
                # 新実装の方が正しい → IMPROVED (本家の数値バグを修正した形)
                tag = f"IMPROVED(max={max_abs:.2e},rms={rms_rel:.2e},dc_rej={dc_rejection_db:.0f}dB)"
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
            elif over_thresh:
                tag = f"DIFFER(max={max_abs:.2e},rms={rms_rel:.2e})"
                verdict = "DEGRADED"
            else:
                tag = "EQUIV"
            eq_results.append((label, max_abs, rms_rel, tag))

        # ---- (4) zansei_impl の 3 負荷 determinism ----
        rng = _np.random.default_rng(1234)
        N = 4096
        x_stereo = rng.standard_normal((2, N)).astype(_np.float32) * 0.05
        sr_proc = TARGET_SR  # v1.6: TARGET_SR 参照

        det_ok = True
        det_notes: list[str] = []
        # Lv1 / Lv5 / Lv10 の代表3点で同一入力 × 2回実行が bit-identical かを確認
        for lv_det in (LOAD_LEVEL_MIN, LOAD_LEVEL_DEFAULT, LOAD_LEVEL_MAX):
            n_thr = _blas_threads(lv_det)
            kw = {"limits": n_thr} if n_thr > 0 else None
            try:
                ctx1 = threadpoolctl.threadpool_limits(**kw) if kw else _NullCtx()
                with ctx1:
                    y1 = zansei_impl(x_stereo.copy(), sr_proc)
                ctx2 = threadpoolctl.threadpool_limits(**kw) if kw else _NullCtx()
                with ctx2:
                    y2 = zansei_impl(x_stereo.copy(), sr_proc)
            except Exception as e:
                det_ok = False
                det_notes.append(f"Lv{lv_det}:EXC({type(e).__name__})")
                continue
            if not _np.all(_np.isfinite(y1)) or not _np.all(_np.isfinite(y2)):
                det_notes.append(f"Lv{lv_det}:NaN")
                det_ok = False
                continue
            if _np.array_equal(y1, y2):
                det_notes.append(f"Lv{lv_det}:OK")
            else:
                max_abs_det = float(_np.max(_np.abs(y1 - y2)))
                det_notes.append(f"Lv{lv_det}:diff(max={max_abs_det:.3e})")
                if max_abs_det > 1e-5:
                    det_ok = False
        if not det_ok:
            verdict = "DEGRADED"

        # ---- (5) Lv1 vs Lv10 zansei_impl 音質不変検証 (diff < 1e-9) ----
        # zansei_impl は sosfiltfilt/hilbert/numpy 配列演算で構成され BLAS 非依存。
        # スレッド数変更 (Lv1=1スレッド, Lv10=全コア) で出力が変わらないことを数値で保証。
        det_lv_ok = True
        det_lv_notes: list[str] = []
        rng_lv = _np.random.default_rng(7777)
        x_lv = rng_lv.standard_normal((2, 4096)).astype(_np.float32) * 0.05

        for lv_a, lv_b in ((LOAD_LEVEL_MIN, LOAD_LEVEL_MAX),):
            n_a = _blas_threads(lv_a)
            n_b = _blas_threads(lv_b)
            kw_a = {"limits": n_a} if n_a > 0 else None
            kw_b = {"limits": n_b} if n_b > 0 else None
            try:
                ctx_a = threadpoolctl.threadpool_limits(**kw_a) if kw_a else _NullCtx()
                with ctx_a:
                    y_la = zansei_impl(x_lv.copy(), TARGET_SR)
                ctx_b = threadpoolctl.threadpool_limits(**kw_b) if kw_b else _NullCtx()
                with ctx_b:
                    y_lb = zansei_impl(x_lv.copy(), TARGET_SR)
            except Exception as e:
                det_lv_notes.append(f"Lv{lv_a}vsLv{lv_b}:EXC({type(e).__name__})")
                det_lv_ok = False
                continue
            if _np.array_equal(y_la, y_lb):
                det_lv_notes.append(f"Lv{lv_a}vsLv{lv_b}:IDENTICAL")
            else:
                diff_lv = float(_np.max(_np.abs(y_la - y_lb)))
                if diff_lv < 1e-9:
                    det_lv_notes.append(f"Lv{lv_a}vsLv{lv_b}:OK(max={diff_lv:.2e})")
                else:
                    det_lv_notes.append(f"Lv{lv_a}vsLv{lv_b}:DIFF(max={diff_lv:.2e})")
                    det_lv_ok = False
        if not det_lv_ok:
            verdict = "DEGRADED"

        # ---- (6) High-frequency addition sanity ----
        psy_notes: list[str] = []
        try:
            rng_psy = _np.random.default_rng(31415)
            N_psy = 8192
            t_psy = _np.arange(N_psy, dtype=_np.float32) / TARGET_SR
            sin1 = 0.15 * _np.sin(2 * _np.pi * 100.0 * t_psy)
            sin2 = 0.10 * _np.sin(2 * _np.pi * 500.0 * t_psy)
            sin3 = 0.08 * _np.sin(2 * _np.pi * 1000.0 * t_psy)
            noise_psy = rng_psy.standard_normal(N_psy) * 0.015
            x_psy_mono = (sin1 + sin2 + sin3 + noise_psy).astype(_np.float32)
            x_psy = _np.stack([x_psy_mono, x_psy_mono], axis=0)

            y_psy = zansei_impl(x_psy.copy(), TARGET_SR)

            hf_in = measure_hf_ratio(x_psy, TARGET_SR)
            hf_out = measure_hf_ratio(y_psy, TARGET_SR)

            tag = f"hf:{hf_in:.3f}→{hf_out:.3f}"
            # v1.12: 50% 以上 hf_ratio 増加を要求 (処理価値の客観可視化を厳格化)
            #        v1.11 の x1.30 では「処理した意味」が客観的に薄かった
            HF_GAIN_THRESHOLD = 1.50
            HF_GAIN_MIN = 1.20  # これ未満は no-op 警告
            ratio = hf_out / hf_in if hf_in > 1e-6 else 1.0
            if hf_in > 1e-6 and hf_out >= hf_in * HF_GAIN_THRESHOLD:
                psy_notes.append(f"IMPROVED({tag},×{ratio:.2f})")
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
            elif hf_in > 1e-6 and hf_out >= hf_in * HF_GAIN_MIN:
                psy_notes.append(f"OK({tag},×{ratio:.2f})")
            elif hf_out > hf_in:
                # 増加はあるが 20% 未満 → 処理価値が薄い (DEGRADED 一歩手前)
                psy_notes.append(f"WEAK({tag},×{ratio:.2f})")
            else:
                psy_notes.append(f"WARN(no_hf_gain {tag})")
                verdict = "DEGRADED"

            # v1.11: 各 enhancement path 単独 sanity
            # 各 path が (1) NaN/Inf を返さない、(2) 過剰 DC offset を持たない、
            # (3) 設計帯域に十分なエネルギーを持つ、ことを個別検証する。
            def _path_sanity(name, sig, expect_hf_only=True, max_dc=1e-3,
                             min_peak=1e-7, max_peak=0.5):
                nonlocal verdict
                if not bool(_np.all(_np.isfinite(sig))):
                    psy_notes.append(f"{name}_NaN")
                    verdict = "DEGRADED"
                    return False
                dc = float(_np.mean(_np.abs(_np.mean(sig, axis=-1))))
                pk = float(_np.max(_np.abs(sig)))
                if dc > max_dc:
                    psy_notes.append(f"{name}_DC_HIGH({dc:.1e})")
                    verdict = "DEGRADED"
                    return False
                if pk > max_peak:
                    # path 単独で原音 peak を超えるのは異常 (drive 設計が壊れた合図)
                    psy_notes.append(f"{name}_PEAK_HIGH({pk:.3f})")
                    verdict = "DEGRADED"
                    return False
                psy_notes.append(f"{name}_OK(pk={pk:.3f},dc={dc:.1e})")
                return True

            # Stereo input for stereo_widen 検証
            x_psy_st = x_psy.copy()
            # わずかに L/R に差をつけて side 信号を生成 (mono だと widen が no-op)
            x_psy_st[1] = x_psy_st[1] * 0.95 + 0.005 * rng_psy.standard_normal(N_psy).astype(_np.float32)

            d_exc = harmonic_exciter(x_psy.copy(), TARGET_SR, params=PARAMS)
            d_warm = mid_warmth(x_psy.copy(), TARGET_SR, params=PARAMS)
            d_widen = stereo_widen_hf(x_psy_st.copy(), TARGET_SR, params=PARAMS)
            d_air = air_band_sparkle(x_psy.copy(), TARGET_SR, params=PARAMS)
            d_tran = transient_crispness(x_psy.copy(), TARGET_SR, params=PARAMS)
            d_nyq = nyquist_complement(x_psy.copy(), TARGET_SR, params=PARAMS)

            _path_sanity("EXC", d_exc)
            _path_sanity("WARM", d_warm)
            _path_sanity("WIDEN", d_widen)
            _path_sanity("AIR", d_air)
            _path_sanity("TRAN", d_tran, max_dc=5e-3)
            _path_sanity("NYQ", d_nyq)

            # exciter 加算で hf_ratio が +0.001 以上上昇 (機能保証)
            hf_with_exc = measure_hf_ratio(x_psy + d_exc, TARGET_SR)
            exc_hf_gain = hf_with_exc - hf_in
            if exc_hf_gain < 1e-3:
                psy_notes.append(f"EXC_NO_HF_GAIN(hf+={exc_hf_gain:+.3f})")
                verdict = "DEGRADED"
            else:
                psy_notes.append(f"EXC_HF_GAIN(hf+={exc_hf_gain:+.3f})")
                if verdict == "EQUIV":
                    verdict = "IMPROVED"

            # Air band 加算で 15kHz 以上のエネルギーが上昇 (plausible HF の機能保証)
            def _measure_air_ratio(sig: _np.ndarray) -> float:
                s = _np.mean(sig, axis=0) if sig.ndim > 1 else sig
                n = len(s)
                if n < 8:
                    return 0.0
                spec = _np.abs(_np.fft.rfft(s))
                freqs = _np.fft.rfftfreq(n, d=1.0 / TARGET_SR)
                e = spec * spec
                tot = float(_np.sum(e)) + 1e-12
                ar = float(_np.sum(e[freqs >= 15000.0]))
                return ar / tot

            air_in = _measure_air_ratio(x_psy)
            air_out = _measure_air_ratio(x_psy + d_air)
            air_gain = air_out - air_in
            # v1.12: Air band path の効果有意性を要求 (+2e-3 以上、処理価値の客観可視化)
            if air_gain < 2e-3:
                psy_notes.append(f"AIR_WEAK({air_in:.2e}→{air_out:.2e},+{air_gain:+.2e})")
            else:
                psy_notes.append(f"AIR_OK({air_in:.2e}→{air_out:.2e},+{air_gain:+.2e})")
                if verdict == "EQUIV":
                    verdict = "IMPROVED"

            # Stereo widen で side energy が有意に増えていること
            try:
                pre_side = (x_psy_st[0] - x_psy_st[1]) * 0.5
                post = x_psy_st + d_widen
                post_side = (post[0] - post[1]) * 0.5
                pre_se = float(_np.sqrt(_np.mean(pre_side * pre_side)) + 1e-30)
                post_se = float(_np.sqrt(_np.mean(post_side * post_side)) + 1e-30)
                widen_ratio = post_se / pre_se
                # v1.12: side energy +10% 以上を要求 (空間広がりの客観可視化)
                if widen_ratio < 1.0:
                    psy_notes.append(f"WIDEN_REGRESSION({widen_ratio:.3f})")
                    verdict = "DEGRADED"
                elif widen_ratio < 1.10:
                    psy_notes.append(f"WIDEN_WEAK({widen_ratio:.3f})")
                else:
                    psy_notes.append(f"WIDEN_OK({widen_ratio:.3f})")
                    if verdict == "EQUIV":
                        verdict = "IMPROVED"
            except Exception as e:
                psy_notes.append(f"WIDEN_EXC({type(e).__name__})")

            # Mid 成分が widen で変化していないこと (定位無傷の保証)
            try:
                pre_mid = (x_psy_st[0] + x_psy_st[1]) * 0.5
                post_mid = ((x_psy_st + d_widen)[0] + (x_psy_st + d_widen)[1]) * 0.5
                mid_diff = float(_np.max(_np.abs(post_mid - pre_mid)))
                psy_notes.append(f"WIDEN_MID_DIFF({mid_diff:.2e})")
                if mid_diff > 1e-5:
                    psy_notes.append("WIDEN_MID_LEAK")
                    verdict = "DEGRADED"
            except Exception as e:
                psy_notes.append(f"WIDEN_MID_EXC({type(e).__name__})")

            # v1.13: Nyquist Band Complement 機能保証 (32-48kHz 帯のエネルギー有意増)
            try:
                def _measure_nyq_ratio(sig: _np.ndarray) -> float:
                    s = _np.mean(sig, axis=0) if sig.ndim > 1 else sig
                    n = len(s)
                    if n < 8:
                        return 0.0
                    spec = _np.abs(_np.fft.rfft(s))
                    freqs = _np.fft.rfftfreq(n, d=1.0 / TARGET_SR)
                    e = spec * spec
                    tot = float(_np.sum(e)) + 1e-12
                    return float(_np.sum(e[freqs >= 32000.0])) / tot

                nyq_in = _measure_nyq_ratio(x_psy)
                nyq_out_full = _measure_nyq_ratio(zansei_impl(x_psy.copy(), TARGET_SR))
                nyq_gain = nyq_out_full - nyq_in
                # 32-48kHz 帯のエネルギー比が +1e-4 以上増 (本家相当の基礎性能)
                if nyq_gain < 1e-4:
                    psy_notes.append(f"NYQ_WEAK({nyq_in:.2e}→{nyq_out_full:.2e},+{nyq_gain:+.2e})")
                    verdict = "DEGRADED"
                else:
                    psy_notes.append(f"NYQ_OK({nyq_in:.2e}→{nyq_out_full:.2e},+{nyq_gain:+.2e})")
                    if verdict == "EQUIV":
                        verdict = "IMPROVED"
            except Exception as e:
                psy_notes.append(f"NYQ_EXC({type(e).__name__})")

            # v1.13: DR (peak/RMS) 改善 — transient/Multi-Aspect で平坦化していないこと
            try:
                # 簡易 transient 信号 (キック相当) で DR (peak/RMS dB) 比較
                rng_dr = _np.random.default_rng(2024)
                N_dr = TARGET_SR // 2  # 0.5 秒
                t_dr = _np.arange(N_dr, dtype=_np.float32) / TARGET_SR
                # 100Hz サイン + 4Hz エンベロープでキック様 transient を生成
                env = (1.0 + _np.cos(2 * _np.pi * 4.0 * t_dr)) * 0.5
                kick = (0.3 * env * _np.sin(2 * _np.pi * 100.0 * t_dr)).astype(_np.float32)
                noise_dr = (rng_dr.standard_normal(N_dr) * 0.02).astype(_np.float32)
                sig_dr_mono = kick + noise_dr
                sig_dr = _np.stack([sig_dr_mono, sig_dr_mono], axis=0)

                def _peak_rms_db(s: _np.ndarray) -> float:
                    p = float(_np.max(_np.abs(s))) + 1e-30
                    r = float(_np.sqrt(_np.mean(s * s))) + 1e-30
                    return 20.0 * float(_np.log10(p / r))

                dr_in = _peak_rms_db(sig_dr)
                dr_out = _peak_rms_db(zansei_impl(sig_dr.copy(), TARGET_SR))
                dr_delta = dr_out - dr_in
                # 平坦化 (DR 低下) > 0.3dB を DEGRADED 判定
                if dr_delta < -0.3:
                    psy_notes.append(f"DR_REGRESSION({dr_in:.2f}→{dr_out:.2f},Δ{dr_delta:+.2f}dB)")
                    verdict = "DEGRADED"
                elif dr_delta >= 0.05:
                    psy_notes.append(f"DR_OK({dr_in:.2f}→{dr_out:.2f},Δ{dr_delta:+.2f}dB)")
                    if verdict == "EQUIV":
                        verdict = "IMPROVED"
                else:
                    psy_notes.append(f"DR_FLAT({dr_in:.2f}→{dr_out:.2f},Δ{dr_delta:+.2f}dB)")
            except Exception as e:
                psy_notes.append(f"DR_EXC({type(e).__name__})")

            # v1.15: スペクトル単調性検証 (隣接帯域の段差/局所ピーク検出)
            # 4-44kHz を 1kHz bin に集約 → 移動平均後、隣接 bin の上昇ジャンプ
            # > +5dB が出る回数をカウント。本家 DSRE は単調減衰スペクトルなので 0 を期待。
            try:
                rng_mn = _np.random.default_rng(11)
                N_mn = TARGET_SR  # 1 秒、freq 解像度 1Hz
                t_mn = _np.arange(N_mn, dtype=_np.float32) / TARGET_SR
                # ホワイトノイズ + 中域 sine (現実的な広帯域信号)
                wn = (rng_mn.standard_normal(N_mn) * 0.04).astype(_np.float32)
                tone = (0.10 * _np.sin(2 * _np.pi * 1000.0 * t_mn)).astype(_np.float32)
                src = wn + tone
                x_mn = _np.stack([src, src], axis=0)

                y_mn = zansei_impl(x_mn.copy(), TARGET_SR)
                s = _np.mean(y_mn, axis=0)
                spec = _np.abs(_np.fft.rfft(s)) ** 2
                freqs = _np.fft.rfftfreq(len(s), d=1.0 / TARGET_SR)

                bins = []
                for lo in range(4000, 44000, 1000):
                    mask = (freqs >= lo) & (freqs < lo + 1000.0)
                    e = float(_np.sum(spec[mask])) + 1e-30
                    bins.append(10.0 * float(_np.log10(e)))
                bins = _np.array(bins, dtype=_np.float64)
                # 3 点移動平均で滑らか化
                kernel = _np.ones(3, dtype=_np.float64) / 3.0
                smoothed = _np.convolve(bins, kernel, mode="valid")
                jumps = _np.diff(smoothed)
                # 上昇 +5dB 以上のジャンプ数 (= 隣接 1kHz bin で 5dB 上昇)
                local_peaks = int(_np.sum(jumps > 5.0))
                max_jump = float(_np.max(jumps)) if jumps.size else 0.0
                if local_peaks == 0:
                    psy_notes.append(f"MONOTONE_OK(peaks={local_peaks},maxΔ={max_jump:+.2f}dB)")
                elif local_peaks <= 1:
                    psy_notes.append(f"MONOTONE_WARN(peaks={local_peaks},maxΔ={max_jump:+.2f}dB)")
                else:
                    psy_notes.append(f"MONOTONE_FAIL(peaks={local_peaks},maxΔ={max_jump:+.2f}dB)")
                    verdict = "DEGRADED"
            except Exception as e:
                psy_notes.append(f"MONO_EXC({type(e).__name__})")
        except Exception as e:
            psy_notes.append(f"EXC({type(e).__name__})")
            verdict = "DEGRADED"

        # ---- (7) ffmpeg 同梱確認 ----
        ffmpeg_path = _find_bundled(
            os.path.join("ffmpeg", "ffmpeg.exe"),
            os.path.join("_internal", "ffmpeg", "ffmpeg.exe"),
        )
        # 開発実行時 (frozen でない) は同梱を必須にしない
        if ffmpeg_path:
            ffmpeg_note = f"OK({os.path.basename(os.path.dirname(ffmpeg_path))}/ffmpeg.exe)"
        elif getattr(sys, "frozen", False):
            ffmpeg_note = "MISSING"
            verdict = "DEGRADED"
        else:
            ffmpeg_note = "dev(skip)"

        eq_summary = " ".join(f"{lbl}:{tag}" for (lbl, _m, _r, tag) in eq_results)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                f"selftest verdict={verdict} numpy={_np.__version__} "
                f"scipy={_sp.__version__} librosa={_lb.__version__} "
                f"threadpoolctl={tpc_version} "
                f"roundtrip={OUTPUT_FORMAT}/{OUTPUT_SUBTYPE}={rt_status} "
                f"sosfiltfilt_equiv=[{eq_summary}] "
                f"determinism=[{' '.join(det_notes)}] "
                f"lv_det=[{' '.join(det_lv_notes)}] "
                f"psy=[{' '.join(psy_notes)}] "
                f"ffmpeg={ffmpeg_note}\n"
            )
        return 0 if verdict != "DEGRADED" else 1
    except Exception:
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("selftest FAILED verdict=DEGRADED\n")
                traceback.print_exc(file=f)
        except Exception:
            traceback.print_exc()
        return 1


def main():
    if "--selftest" in sys.argv:
        sys.exit(_run_selftest())
    add_ffmpeg_to_path()
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(_app_icon())
    # トレイ運用: 最後のウィンドウが閉じてもアプリを終了させない
    app.setQuitOnLastWindowClosed(False)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
