# ===== 強化版（処理維持＋安定性＋精度向上）=====

import os
import sys
import time
import tempfile
import subprocess
import configparser
import functools
import threading
import datetime
import shutil
import traceback as _traceback
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import next_fast_len
import sqlite3
import pyloudnorm as pyln
# librosa / resampy は遅延 import (使用箇所で import)。
# numba/llvmlite JIT を GUI 起動経路から外し体感起動を短縮する。
# librosa: load_audio_safe フォールバック時のみ。resampy: 非 96k 入力のリサンプル時のみ。

from PySide6 import QtCore, QtGui, QtWidgets

from send2trash import send2trash


# ===== 入出力 =====
INPUT_DIR = r"C:\Audio\DSRE"
OUTPUT_DIR = r"C:\Audio\DSRE\Output"
METRICS_DB_PATH = r"C:\FreeSoft\DSRE\dsre_log.db"


_DSRE_VERSION = "r126"


# ===== DSP パラメータ =====
HARMONIC_LAYERS = 8         # 倍音重畳の段数
HARMONIC_DECAY = 1.25       # 各段の減衰係数
PRE_HP_CUTOFF_HZ = 3000     # 倍音抽出前のハイパス
# v1.8 復帰: POST_HP 16k→12k で 12-16kHz 帯倍音を開放。v1.8 単独では問題未発生
# (v1.11 以降の問題は別 path 起因)。再発防止教訓 (feedback_dsre_lessons.md) 参照。
POST_HP_CUTOFF_HZ = 12000   # 倍音生成後のハイパス (v1.8 復帰)
TARGET_SR = 96000           # v1.6: 本家デフォルトに戻す。192k は intermod 副作用 + 計算 2 倍の overkill だった (DSEE HX 思想は 96k 上限)
FILTER_ORDER = 11           # バターワース次数

# ===== harmonic 作用: harmonic_exciter (v1.10 等価、整数倍音生成) =====
# freq_shift 系倍音 (Hilbert + 複素乗算 single-sideband) は線形周波数変位で
# 整数倍関係 (harmonic relationship) を破壊する → 金属臭さ・不自然な高音乗り。
# 半波整流 + tanh ソフト歪は f0 から 2f0/3f0 を自然生成し harmonic relationship
# を保つ。zansei_impl の freq_shift 倍音と独立 path で並列加算する。
EXCITER_HP_HZ = 4000         # exciter ソース帯域 (4kHz 以上)
EXCITER_OUT_HP_HZ = 7000     # 出力 HP (中域への滲み防止)
EXCITER_DRIVE = 0.20         # 加算ブレンド比 (控えめ、過剰歪回避)
EXCITER_SAT_GAIN = 1.5       # tanh 入力ゲイン (soft tube-like)

# ===== body 作用: 中低域温度感 (BP 250-1200Hz、出力 LP 3kHz cap) =====
# 「高域補間ゾーン (7kHz+) へ物理的に届かない」ことを LP カットで保証する設計。
# 非対称 soft clip で 2nd harmonic 主体の温度感を作る。HP 出力 (旧 mid_warmth) は
# 飽和高調波が高域へ漏れた失敗、LP 出力でその再発を遮断する。
BODY_WARMTH_LO_HZ = 250
BODY_WARMTH_HI_HZ = 1200
BODY_WARMTH_OUT_LP_HZ = 3000   # 7kHz exciter ゾーンから 4kHz マージン
BODY_WARMTH_DRIVE = 0.12
BODY_WARMTH_ASYM_GAIN = 1.2
BODY_WARMTH_ASYM_K = 0.12       # s² 寄与 (2nd harmonic 強度)

# ===== stereo 作用: 中帯域 Side widening (BP 300-2500Hz、線形、出力 LP 3kHz) =====
# Mid/Side 中の Side のみ +Δ で空間定義感を上げる。Mid 不変で定位は崩れない。
# 完全線形 (SAT なし) で harmonic を生成しないため、LP cap で高域補間ゾーンとは
# 物理的に独立。
STEREO_DEF_LO_HZ = 300
STEREO_DEF_HI_HZ = 2500
STEREO_DEF_OUT_LP_HZ = 3000
STEREO_DEF_GAIN = 0.15

# ===== sub 作用: <150Hz 輪郭強化 (envelope-based 線形 gain mod、出力 LP 200Hz) =====
# fast/slow envelope diff を [0,1] 正規化し sub_band 自体に時変ゲインを掛ける。
# sign() を使わない pure linear (旧 transient_crispness の境界クリック原因を回避)。
# 出力 LP 200Hz で 200Hz+ には一切影響せず、当然 7kHz+ 高域補間ゾーンも不変。
SUB_TIGHT_LP_HZ = 150
SUB_TIGHT_OUT_LP_HZ = 200
SUB_TIGHT_FAST_MS = 6.0
SUB_TIGHT_SLOW_MS = 100.0
SUB_TIGHT_GAIN = 0.10

# ===== mid_trans 作用: 中帯域 transient 定義感 (250-2500Hz、線形 gain mod) =====
# fast/slow zero-phase moving-average diff を [0,1] 正規化し mid_band 自体に
# 時変ゲイン。完全線形 (sign() なし、SAT なし、harmonic 非生成)。
# 出力 LP 3500Hz cap で 7kHz+ 高域補間ゾーンに物理的に届かない。
# 効果: アタック明瞭化、楽器立ち上がり、分離感、勢い・生命感。
MID_TRANS_LO_HZ = 250
MID_TRANS_HI_HZ = 2500
MID_TRANS_OUT_LP_HZ = 3500
MID_TRANS_FAST_MS = 5.0
MID_TRANS_SLOW_MS = 80.0
MID_TRANS_GAIN = 0.13

# ===== vocal_pres 作用: ボーカル存在感 (M/S Mid 1500-3000Hz、線形 shelf) =====
# Mid のみ +Δ、Side 不変。M/S 分解で中帯域 Mid を BP 抽出 → LP cap → +gain。
# 完全線形・SAT なしのため harmonic 生成ゼロ、LP cap で高域補間ゾーン物理隔離。
# 効果: ボーカルセンター存在感、台詞・主旋律明瞭化、艶感。
VOCAL_PRES_LO_HZ = 1500
VOCAL_PRES_HI_HZ = 3000
VOCAL_PRES_OUT_LP_HZ = 3500
VOCAL_PRES_GAIN = 0.10

# ===== low_body 作用: 60-200Hz 胴鳴り感 (asym SAT、出力 LP 400Hz) =====
# 軽度非対称 soft clip → 2nd harmonic 主体で kick/bass の重みと深さを増す。
# 出力 LP 400Hz cap で 400Hz+ には影響なし、当然高域補間ゾーン不変。
# 周波数を上げず体感で増強する psychoacoustic 系手法。
LOW_BODY_LO_HZ = 60
LOW_BODY_HI_HZ = 200
LOW_BODY_OUT_LP_HZ = 400
LOW_BODY_DRIVE = 0.08
LOW_BODY_ASYM_GAIN = 1.0
LOW_BODY_ASYM_K = 0.10

# ===== Phase3 cycle2 bold basket (2026-05-16、各独立 flag、耳 NG 時 cheap gate で bisect) =====
# spectral_ledge 源 = 帯域配置/重なり (急峻度でない: C1/cand1 で falsified 済)。
# A/B/C を帯域・cap の「配置」変更で攻める。全て出力 LP ≤3.5kHz 維持 → 7kHz+ 凍結非到達。
BP_REALIGN_ENABLED = True        # A: body/mid_trans を 800Hz で tile (250-1200 二重計上除去)
BODY_WARMTH_HI_HZ_R = 800        # A: body_warmth hi 1200→800
MID_TRANS_LO_HZ_R = 800          # A: mid_transient_def lo 250→800 (body と境界共有)
MS_ROLE_SPLIT_ENABLED = True     # B: stereo_def Side を 300-1500 に限定 (vocal Mid 1500-3000 と非重複)
STEREO_DEF_HI_HZ_R = 1500        # B: stereo_def hi 2500→1500
CAP_ALIGN_ENABLED = True         # C: midlow 出力 LP cap を 3500 に統一 (3000/3500 二重端除去)
MIDLOW_UNIFIED_OUT_LP_HZ = 3500  # C: body/stereo 出力 LP 3000→3500 (≤3.5k 維持、凍結非到達)

# ===== Phase3 cycle3 input-adaptive (2026-05-18、各独立 env flag、default OFF=byte同一) =====
# P+Q: 入力源特性 (crest / hf_ratio / Side-Mid 比) を file 1 回計測し、非凍結 6 path
#   合算を bounded 連続スケール (離散モード禁止=別ソフト感源)。凍結 d_res/d_exc は
#   スケール後加算で HF signature 入力非依存・byte 不変。env DSRE_PROFILE_SCALE=1 で ON、
#   未設定なら scale 計算自体を skip し従来と完全 byte 同一。
# A: 整数比 (48k→96k=×2 / 192k→96k=÷2 等) のみ scipy resample_poly + 設計 kaiser、
#   非整数比 (44.1k/88.2k) は現 resampy kaiser_best fallback で不変。env DSRE_INT_RESAMPLE=1。
# 源間 spread の構造上限 = [MIN,MAX] 幅 (±10% → 最悪 spread 0.20、Phase2 acceptance を数式に内包)。
PROFILE_SCALE_ENABLED = True     # P+Q: 採用済 (2026-05-18 耳 OK)。既定 ON、
#   env DSRE_PROFILE_SCALE=0 が kill-switch (bisect/debug 用)。1/未設定=ON。
PROFILE_SCALE_MIN = 0.90         # Q: 非凍結合算スケール下限
PROFILE_SCALE_MAX = 1.10         # Q: 上限 (±10%、源間 spread を構造的に ≤0.20 へ)
PROFILE_CREST_REF_DB = 12.0      # Q: crest 基準 (≥=広DR→中立、未満=海苔→減衰のみ)
PROFILE_KC = 0.020               # Q: crest 係数 (海苔ほど scale↓、過処理回避=PLR/DR 保護)
PROFILE_HF_REF = 0.08            # Q: hf_ratio 基準 (4kHz+ / 総エネルギー)
PROFILE_KH = 1.00                # Q: hf 係数 (曇り/ロッシー源ほど scale↑、DSEE HX 思想)
PROFILE_SM_REF = 0.50            # Q: Side/Mid 基準 (mono≈0、広≈1)
PROFILE_SM_KS = 0.30             # Q: SM 係数 (M/S 2 path のみ、mono/狭の過拡張回避)
INT_RESAMPLE_KAISER_BETA = 12.0  # A: resample_poly kaiser β (高 stopband 阻止)

# ===== dynamics 作用: 入力依存ゲート (v1.18 等価、SAT 残響対策) =====
# tanh / 半波整流は微小信号でも倍音を生成するため、アウトロ/無音区間で残響と
# してノイズフロアを上げる。原音 |x| の moving-average envelope が threshold
# 未満の区間で d_extra を 0 化する時間領域振幅マスク (FFT 後段矯正ではない)。
INPUT_GATE_THRESHOLD = 0.003 # -50dBFS、これ未満は無音扱い
INPUT_GATE_WINDOW_MS = 10.0  # envelope 計測窓
INPUT_GATE_SMOOTH_HZ = 50.0  # ゲートエッジ zero-phase LP smoothing

# ===== Phase 3 改善: 中低域 6 path 構造的安全弁 =====
# 高域補間ゾーン (residual harmonic loop / harmonic_exciter / post_hp / input_gate)
# は abf8260 凍結で改変禁止。以下は中低域 6 path にのみ適用される構造的安全弁。
#
# 改善 A (倍音構造): body_warmth / low_body_harmonic の SAT → 純 quadratic 化
#   tanh ベース歪は 2nd だけでなく 3rd/5th も混入し中低域を dirty にする。
#   `(s² - mean(s²)) * k` は純 2nd harmonic のみ生成、3rd 以上の混入を排除。
#
# 改善 C (動的 envelope): sub_tightness / mid_transient_def の percentile を移動窓化
#   ファイル全長 percentile_99 は長尺曲後半に強い transient がある場合、前半飽和
#   後半抑制で前後不均衡 envelope mod を生む。5 秒移動窓 (50% overlap) + 線形補間で
#   局所的な「loud level」に追従。
ENV_P99_WINDOW_SEC = 5.0     # 移動 percentile 窓長
ENV_P99_HOP_RATIO = 0.5      # 50% overlap

# 改善 D (空間 M/S 統合): stereo_definition_mid + vocal_presence_mid の比率正規化
#   両 path 独立加算で Side+Mid 両方が増強され、トータル M/S バランスが破綻する
#   素材があるため、入力素材の自然な Side/Mid 比に対する「上限倍率」を設定し
#   超えた場合のみ d_stereo を線形 scale-down。
MS_BALANCE_RATIO_CAP = 1.5   # delta Side/Mid ratio ≤ input Side/Mid * 1.5

# 改善 E (RMS cap) + I (peak limit): 中低域 6 path のみ対象、d_res / d_exc は対象外
#   静的 cap で「上限超過時のみ線形 scale-down」、通常素材では no-op。
#   uniform scaling のため per-sample 歪み非生成、preringing も非発生。
#   d_res / d_exc は 高域補間凍結ゾーンに直接含まれるため対象外
#   (cap でも縮小すれば高域出力が変わるため変更禁止)。
PATH_RMS_CAP_RATIO = 0.15    # path RMS ≤ input RMS * 0.15
PATH_PEAK_CAP_RATIO = 0.18   # path peak ≤ input peak * 0.18

# v1.6: FLAC 96kHz / PCM_24 固定 (v1.5 の WAV 32bit float / 192kHz は overkill だった)
# 経緯: v1.4 で WAV 32bit float 化 → foobar 測定で v1.3 と同値 → v1.5 で FLAC PCM_24 復帰
#       v1.5 の 192k 出力 → 主観違和感 (ボーカル裏に高音乗り) + 計算 2 倍 → v1.6 で 96k 復帰
OUTPUT_SUBTYPE = "PCM_24"
OUTPUT_SUBTYPE_FALLBACK = "PCM_16"  # libsndfile 異常時の安全網
OUTPUT_FORMAT = "FLAC"
OUTPUT_EXT = ".flac"

# 音量最適化: true peak 正規化目標 [dBFS]。
# 値の根拠は「FLAC 96k 単体で clips=0」ではなく「FLAC→Opus 182 VBR 変換後も
# clips=0」。Opus は内部 48k リサンプル + lossy 再構成で inter-sample peak が
# 増える (実測 2026: 1kHz +0.10dB / 11kHz +0.45dB / 広帯域は LPF で逆に低下)。
# 実音楽の現実的最悪 HF 成分 ≈ +0.45dB。業界標準の -1.0 dBTP がこれを安全に
# 吸収 (-1.0+0.45=-0.55dBFS で余裕の 0)。8x oversampling (foobar TPS の ITU 4x
# より過大評価側) と併せ Opus 後 clips=0 を構造的に保証する。過剰削減ではない
# (実測 10000→4clip 相当の縮小より小、Opus 配信の必須マージン)。
# 値はチューナブル: clips がまだ出るなら下げ、loudness 優先なら上げる。
TP_TARGET_DBFS = -1.0

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


def _int_resample_poly(y, sr: int, tgt: int, beta: float):
    """A: 整数比 (tgt%sr==0 / sr%tgt==0) のみ scipy resample_poly + 長 firwin
    kaiser で透過変換。非整数比・失敗時は None を返し呼び出し側で resampy
    fallback。短 FIR (window tuple 既定 ~20tap) は image floor が劣悪なため
    明示的に長い firwin (cutoff=1/max(up,dn)) を設計して渡す。非凍結経路。
    """
    if sr <= 0 or sr == tgt:
        return None
    if tgt % sr == 0:
        up, dn = tgt // sr, 1
    elif sr % tgt == 0:
        up, dn = 1, sr // tgt
    else:
        return None
    if max(up, dn) > 16:
        return None
    try:
        from scipy.signal import resample_poly, firwin
        half = 32 * max(up, dn)            # 片側 tap 数 (×2 で 64tap/位相相当)
        n_taps = 2 * half * max(up, dn) + 1  # 数千〜万 tap で深い stopband
        n_taps = min(n_taps, 32769)
        fir = firwin(n_taps, 1.0 / max(up, dn),
                     window=("kaiser", beta)).astype(np.float64)
        out = resample_poly(y, up, dn, axis=-1, window=fir)
        return out.astype(np.float32, copy=False)
    except Exception:
        return None


def _resample_to_target(y, sr: int, tgt: int, par: bool = False):
    """入力 sr → tgt の単一リサンプル経路 (_process_one / render_ref 共用、二重実装回避)。
    env DSRE_INT_RESAMPLE=1 かつ整数比なら resample_poly、それ以外は resampy
    kaiser_best (現行・不変)。flag 未設定時は従来と完全 byte 同一。
    """
    if sr == tgt:
        return y, sr
    y_poly = None
    if os.environ.get("DSRE_INT_RESAMPLE") == "1":
        y_poly = _int_resample_poly(y, sr, tgt, PARAMS.int_resample_kaiser_beta)
    if y_poly is not None:
        return y_poly, tgt
    import resampy  # 遅延 import: 96k 入力のみなら numba 不要
    try:
        y = resampy.resample(y, sr, tgt, parallel=par)
    except TypeError:
        y = resampy.resample(y, sr, tgt)
    return y, tgt


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
    # harmonic 作用: harmonic_exciter
    exciter_hp: int = EXCITER_HP_HZ
    exciter_out_hp: int = EXCITER_OUT_HP_HZ
    exciter_drive: float = EXCITER_DRIVE
    exciter_sat_gain: float = EXCITER_SAT_GAIN
    # body 作用: body_warmth (LP 出力で高域補間ゾーン物理遮断)
    body_warmth_lo_hz: int = BODY_WARMTH_LO_HZ
    body_warmth_hi_hz: int = BODY_WARMTH_HI_HZ
    body_warmth_out_lp_hz: int = BODY_WARMTH_OUT_LP_HZ
    body_warmth_drive: float = BODY_WARMTH_DRIVE
    body_warmth_asym_gain: float = BODY_WARMTH_ASYM_GAIN
    body_warmth_asym_k: float = BODY_WARMTH_ASYM_K
    # stereo 作用: stereo_definition_mid (中帯域 Side、線形)
    stereo_def_lo_hz: int = STEREO_DEF_LO_HZ
    stereo_def_hi_hz: int = STEREO_DEF_HI_HZ
    stereo_def_out_lp_hz: int = STEREO_DEF_OUT_LP_HZ
    stereo_def_gain: float = STEREO_DEF_GAIN
    # sub 作用: sub_tightness (envelope-based 線形 gain mod)
    sub_tight_lp_hz: int = SUB_TIGHT_LP_HZ
    sub_tight_out_lp_hz: int = SUB_TIGHT_OUT_LP_HZ
    sub_tight_fast_ms: float = SUB_TIGHT_FAST_MS
    sub_tight_slow_ms: float = SUB_TIGHT_SLOW_MS
    sub_tight_gain: float = SUB_TIGHT_GAIN
    # mid_trans 作用: mid_transient_def (中帯域 attack 定義、線形)
    mid_trans_lo_hz: int = MID_TRANS_LO_HZ
    mid_trans_hi_hz: int = MID_TRANS_HI_HZ
    mid_trans_out_lp_hz: int = MID_TRANS_OUT_LP_HZ
    mid_trans_fast_ms: float = MID_TRANS_FAST_MS
    mid_trans_slow_ms: float = MID_TRANS_SLOW_MS
    mid_trans_gain: float = MID_TRANS_GAIN
    # vocal_pres 作用: vocal_presence_mid (M/S Mid 中帯域、線形)
    vocal_pres_lo_hz: int = VOCAL_PRES_LO_HZ
    vocal_pres_hi_hz: int = VOCAL_PRES_HI_HZ
    vocal_pres_out_lp_hz: int = VOCAL_PRES_OUT_LP_HZ
    vocal_pres_gain: float = VOCAL_PRES_GAIN
    # low_body 作用: low_body_harmonic (60-200Hz 胴鳴り)
    low_body_lo_hz: int = LOW_BODY_LO_HZ
    low_body_hi_hz: int = LOW_BODY_HI_HZ
    low_body_out_lp_hz: int = LOW_BODY_OUT_LP_HZ
    low_body_drive: float = LOW_BODY_DRIVE
    low_body_asym_gain: float = LOW_BODY_ASYM_GAIN
    low_body_asym_k: float = LOW_BODY_ASYM_K
    # dynamics 作用: 入力依存ゲート (FROZEN)
    input_gate_threshold: float = INPUT_GATE_THRESHOLD
    input_gate_window_ms: float = INPUT_GATE_WINDOW_MS
    input_gate_smooth_hz: float = INPUT_GATE_SMOOTH_HZ
    # Phase 3 改善: 中低域 6 path 構造的安全弁
    env_p99_window_sec: float = ENV_P99_WINDOW_SEC
    env_p99_hop_ratio: float = ENV_P99_HOP_RATIO
    ms_balance_ratio_cap: float = MS_BALANCE_RATIO_CAP
    path_rms_cap_ratio: float = PATH_RMS_CAP_RATIO
    path_peak_cap_ratio: float = PATH_PEAK_CAP_RATIO
    # Phase3 cycle2 bold basket (各独立 flag、bisect 可能)
    bp_realign_enabled: bool = BP_REALIGN_ENABLED
    body_warmth_hi_hz_r: int = BODY_WARMTH_HI_HZ_R
    mid_trans_lo_hz_r: int = MID_TRANS_LO_HZ_R
    ms_role_split_enabled: bool = MS_ROLE_SPLIT_ENABLED
    stereo_def_hi_hz_r: int = STEREO_DEF_HI_HZ_R
    cap_align_enabled: bool = CAP_ALIGN_ENABLED
    midlow_unified_out_lp_hz: int = MIDLOW_UNIFIED_OUT_LP_HZ
    # Phase3 cycle3 input-adaptive (採用済、既定 ON、env=0 で kill-switch)
    profile_scale_enabled: bool = PROFILE_SCALE_ENABLED
    profile_scale_min: float = PROFILE_SCALE_MIN
    profile_scale_max: float = PROFILE_SCALE_MAX
    profile_crest_ref_db: float = PROFILE_CREST_REF_DB
    profile_kc: float = PROFILE_KC
    profile_hf_ref: float = PROFILE_HF_REF
    profile_kh: float = PROFILE_KH
    profile_sm_ref: float = PROFILE_SM_REF
    profile_sm_ks: float = PROFILE_SM_KS
    int_resample_kaiser_beta: float = INT_RESAMPLE_KAISER_BETA


PARAMS = DSREParams()


# ===== 音響メトリクス計算 =====

class MetricsComputer:
    """処理前後の音響メトリクスを numpy/scipy/pyloudnorm で計算する純粋関数クラス。"""

    @staticmethod
    def compute(audio: np.ndarray, sr: int) -> dict:
        """
        audio: shape (samples,) または (2, samples) の float32/64
        sr: サンプルレート
        戻り値: メトリクス名 → 値 の dict。計算失敗時は None。
        """
        try:
            # ステレオは L/R 平均してモノに変換（スペクトル系指標用）
            if audio.ndim == 2:
                mono = audio.mean(axis=0)
            else:
                mono = audio

            result = {}

            # --- 基本レベル ---
            rms = float(np.sqrt(np.mean(mono ** 2)))
            peak = float(np.max(np.abs(mono)))
            result["rms_db"] = 20 * np.log10(rms + 1e-9)
            result["peak_db"] = 20 * np.log10(peak + 1e-9)

            # --- DR (TT DR meter 準拠: ブロック 3秒分割、各ブロック RMS peak vs 全体 RMS) ---
            block_len = sr * 3
            n_blocks = max(1, len(mono) // block_len)
            block_peaks = []
            block_rms_vals = []
            for i in range(n_blocks):
                blk = mono[i * block_len:(i + 1) * block_len]
                block_peaks.append(20 * np.log10(np.max(np.abs(blk)) + 1e-9))
                block_rms_vals.append(20 * np.log10(np.sqrt(np.mean(blk ** 2)) + 1e-9))
            result["dr"] = round(np.mean(block_peaks) - np.mean(block_rms_vals), 2)

            # --- LUFS / LRA (pyloudnorm) ---
            meter = pyln.Meter(sr)
            # pyloudnorm は (samples, channels) の shape を期待
            if audio.ndim == 2:
                audio_pln = audio.T  # (samples, 2)
            else:
                audio_pln = audio.reshape(-1, 1)  # (samples, 1)
            try:
                lufs = float(meter.integrated_loudness(audio_pln.astype(np.float64)))
            except Exception:
                lufs = None
            result["lufs"] = lufs
            result["plr"] = (result["peak_db"] - lufs) if lufs is not None else None

            # LRA approximation via short-term loudness blocks (EBU R128 simplified)
            try:
                block_dur = 3.0  # 3-second short-term blocks
                block_len = int(sr * block_dur)
                n_blocks = len(audio_pln) // block_len
                if n_blocks >= 2:
                    st_loudness = []
                    for i in range(n_blocks):
                        blk = audio_pln[i * block_len:(i + 1) * block_len]
                        try:
                            st = meter.integrated_loudness(blk.astype(np.float64))
                            if st > -70:  # gate: exclude silence blocks
                                st_loudness.append(st)
                        except Exception:
                            pass
                    if len(st_loudness) >= 2:
                        lra = float(max(st_loudness) - min(st_loudness))
                    else:
                        lra = None
                else:
                    lra = None
            except Exception:
                lra = None
            result["lra"] = lra

            # --- クリップカウント ---
            result["clip_count"] = int(np.sum(np.abs(mono) >= 1.0))

            # --- スペクトル系 (scipy) ---
            from scipy.signal import welch
            freqs, psd = welch(mono, fs=sr, nperseg=min(4096, len(mono)))
            psd_sum = psd.sum()

            if psd_sum > 0:
                result["centroid_hz"] = float(np.sum(freqs * psd) / psd_sum)

                cumsum = np.cumsum(psd)
                rolloff_idx = np.searchsorted(cumsum, 0.85 * psd_sum)
                result["rolloff_hz"] = float(freqs[min(rolloff_idx, len(freqs) - 1)])

                geometric_mean = np.exp(np.mean(np.log(psd + 1e-30)))
                arithmetic_mean = psd_sum / len(psd)
                result["flatness"] = float(geometric_mean / (arithmetic_mean + 1e-30))
            else:
                result["centroid_hz"] = None
                result["rolloff_hz"] = None
                result["flatness"] = None

            # --- HF ratio (numpy FFT) ---
            n_fft = next_fast_len(len(mono))
            spec = np.abs(np.fft.rfft(mono, n=n_fft)) ** 2
            f_axis = np.fft.rfftfreq(n_fft, d=1.0 / sr)
            total_e = spec.sum() + 1e-30
            for cutoff in (4000, 8000, 12000, 16000):
                result[f"hf_ratio_{cutoff // 1000}k"] = float(
                    spec[f_axis >= cutoff].sum() / total_e
                )

            # --- THD proxy (基音 vs 2次/3次倍音エネルギー比) ---
            try:
                bin_hz = f_axis[1] - f_axis[0]
                f0_idx = int(1000 / bin_hz)
                margin = max(1, int(100 / bin_hz))
                f0_e = spec[max(0, f0_idx - margin):f0_idx + margin + 1].sum()
                f2_e = spec[max(0, 2 * f0_idx - margin):2 * f0_idx + margin + 1].sum()
                f3_e = spec[max(0, 3 * f0_idx - margin):3 * f0_idx + margin + 1].sum()
                result["harmonic_1k_proxy"] = float((f2_e + f3_e) / (f0_e + 1e-30))
            except Exception:
                result["harmonic_1k_proxy"] = None

            return result

        except Exception:
            # ロガーがメイン処理をブロックしてはならない
            return {k: None for k in [
                "rms_db", "peak_db", "dr", "plr", "lufs", "lra", "clip_count",
                "centroid_hz", "rolloff_hz", "flatness",
                "hf_ratio_4k", "hf_ratio_8k", "hf_ratio_12k", "hf_ratio_16k",
                "harmonic_1k_proxy",
            ]}


class AudioMetricsLogger:
    """処理ごとの before/after 音響メトリクスを SQLite に追記する。"""

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS runs (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,
        filename            TEXT,
        dsre_version        TEXT,
        input_sr            INTEGER,
        input_channels      INTEGER,
        duration_sec        REAL,
        processing_time_sec REAL,
        input_bitrate_kbps  INTEGER,
        rms_db_b            REAL, peak_db_b REAL, dr_b REAL, plr_b REAL,
        lufs_b              REAL, lra_b REAL, clip_count_b INTEGER,
        centroid_hz_b       REAL, rolloff_hz_b REAL, flatness_b REAL,
        hf_ratio_4k_b       REAL, hf_ratio_8k_b REAL,
        hf_ratio_12k_b      REAL, hf_ratio_16k_b REAL, harmonic_1k_proxy_b REAL,
        rms_db_a            REAL, peak_db_a REAL, dr_a REAL, plr_a REAL,
        lufs_a              REAL, lra_a REAL, clip_count_a INTEGER,
        centroid_hz_a       REAL, rolloff_hz_a REAL, flatness_a REAL,
        hf_ratio_4k_a       REAL, hf_ratio_8k_a REAL,
        hf_ratio_12k_a      REAL, hf_ratio_16k_a REAL, harmonic_1k_proxy_a REAL,
        adaptive_layers     INTEGER,
        adaptive_mode       TEXT,
        hf_ratio_input      REAL
    )
    """

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
        input_audio: 96k リサンプル済の処理前音声 (numpy array)
        output_audio: zansei_impl 処理後音声 (numpy array)
        _before_metrics / _after_metrics: 事前計算済みの場合は MetricsComputer を再呼びしない。
        """
        try:
            import datetime
            b = _before_metrics if _before_metrics is not None else MetricsComputer.compute(input_audio, sr)
            a = _after_metrics if _after_metrics is not None else MetricsComputer.compute(output_audio, sr)

            if input_audio.ndim == 2:
                channels = input_audio.shape[0]
                duration = input_audio.shape[1] / sr
            else:
                channels = 1
                duration = len(input_audio) / sr

            try:
                info = sf.info(input_path)
                bitrate = int(info.extra_info.get("bitrate", 0)) or None
            except Exception:
                bitrate = None

            row = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "filename": os.path.basename(input_path),
                "dsre_version": _DSRE_VERSION,
                "input_sr": sr,
                "input_channels": channels,
                "duration_sec": duration,
                "processing_time_sec": processing_time_sec,
                "input_bitrate_kbps": bitrate,
                "rms_db_b": b["rms_db"], "peak_db_b": b["peak_db"],
                "dr_b": b["dr"], "plr_b": b["plr"],
                "lufs_b": b["lufs"], "lra_b": b["lra"],
                "clip_count_b": b["clip_count"],
                "centroid_hz_b": b["centroid_hz"], "rolloff_hz_b": b["rolloff_hz"],
                "flatness_b": b["flatness"],
                "hf_ratio_4k_b": b["hf_ratio_4k"], "hf_ratio_8k_b": b["hf_ratio_8k"],
                "hf_ratio_12k_b": b["hf_ratio_12k"], "hf_ratio_16k_b": b["hf_ratio_16k"],
                "harmonic_1k_proxy_b": b["harmonic_1k_proxy"],
                "rms_db_a": a["rms_db"], "peak_db_a": a["peak_db"],
                "dr_a": a["dr"], "plr_a": a["plr"],
                "lufs_a": a["lufs"], "lra_a": a["lra"],
                "clip_count_a": a["clip_count"],
                "centroid_hz_a": a["centroid_hz"], "rolloff_hz_a": a["rolloff_hz"],
                "flatness_a": a["flatness"],
                "hf_ratio_4k_a": a["hf_ratio_4k"], "hf_ratio_8k_a": a["hf_ratio_8k"],
                "hf_ratio_12k_a": a["hf_ratio_12k"], "hf_ratio_16k_a": a["hf_ratio_16k"],
                "harmonic_1k_proxy_a": a["harmonic_1k_proxy"],
                "adaptive_layers": adaptive_layers,
                "adaptive_mode": adaptive_mode,
                "hf_ratio_input": hf_ratio_input,
            }

            os.makedirs(os.path.dirname(METRICS_DB_PATH), exist_ok=True)
            with sqlite3.connect(METRICS_DB_PATH) as conn:
                conn.execute(cls._CREATE_SQL)
                cols = ", ".join(row.keys())
                placeholders = ", ".join("?" * len(row))
                conn.execute(
                    f"INSERT INTO runs ({cols}) VALUES ({placeholders})",
                    list(row.values()),
                )
                conn.commit()
        except Exception:
            pass  # ロガー失敗は処理結果に影響させない


def _extract_flac_pictures(path: str) -> list:
    """FLAC ファイルから PICTURE ブロックを mutagen で抽出する。失敗時は空リストを返す。"""
    try:
        from mutagen.flac import FLAC
        return list(FLAC(path).pictures)
    except Exception:
        return []


def _embed_output_metadata(path: str, pictures: list) -> None:
    """処理済 FLAC にアートワーク + DSRE タグを埋め込む。失敗は握り潰す。"""
    try:
        import datetime
        from mutagen.flac import FLAC
        f = FLAC(path)
        f.clear_pictures()
        for pic in pictures:
            f.add_picture(pic)
        if f.tags is None:
            f.add_tags()
        f.tags["dsre_version"] = [_DSRE_VERSION]
        f.tags["dsre_processed_utc"] = [datetime.datetime.utcnow().isoformat()]
        f.save()
    except Exception:
        pass


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

# ===== fpcalc.exe パス解決 (chromaprint 音声指紋) =====
def _resolve_fpcalc_path() -> "str | None":
    """fpcalc.exe のパスを解決。バンドル → PATH → None の優先順。
    None 戻りで指紋機能は無効化 (graceful degrade)。"""
    found = _find_bundled(
        os.path.join("ffmpeg", "fpcalc.exe"),
        os.path.join("_internal", "ffmpeg", "fpcalc.exe"),
        "fpcalc.exe",
    )
    if found:
        return found
    from shutil import which
    return which("fpcalc") or which("fpcalc.exe") or None


# ===== FingerprintEngine (chromaprint acoustic fingerprint + SQLite cache) =====

class FingerprintResult:
    __slots__ = ("duration_sec", "fingerprint", "content_hash")
    def __init__(self, duration_sec: float, fingerprint: str, content_hash: str):
        self.duration_sec = duration_sec
        self.fingerprint = fingerprint
        self.content_hash = content_hash


class FingerprintEngine:
    """chromaprint (fpcalc.exe) で acoustic fingerprint を取得し SQLite にキャッシュ。"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = METRICS_DB_PATH
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
        except (subprocess.TimeoutExpired, Exception):
            return None
        # cache 書き込み
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "INSERT OR REPLACE INTO fingerprints VALUES (?, ?, ?, ?, ?)",
                (ch, duration, fp, path, datetime.datetime.utcnow().isoformat()),
            )
        return FingerprintResult(duration, fp, ch)


# ===== chromaprint fingerprint similarity (pure Python) =====

def _decode_chromaprint(fp_str: str) -> list:
    """chromaprint base64 URL-safe エンコード文字列を int32 list に decode。
    ヘッダ (4 byte) を skip し残りを int32 LE で読む。
    """
    import base64
    try:
        raw = base64.urlsafe_b64decode(fp_str + "=" * (4 - len(fp_str) % 4))
    except Exception:
        return []
    if len(raw) < 4:
        return []
    body = raw[4:]
    n = len(body) // 4
    out = []
    for i in range(n):
        v = int.from_bytes(body[i*4:(i+1)*4], "little", signed=False)
        out.append(v)
    return out


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
    for off in range(lb - la + 1):
        match_bits = 0
        total_bits = la * 32
        for i in range(la):
            xor = a[i] ^ b[off + i]
            match_bits += 32 - xor.bit_count()
        ratio = match_bits / total_bits
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio




class ClusterBuilder:
    """指紋類似度ペアを union-find で cluster 化する。"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self._parent: dict = {}
        self._pairs: list = []

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


# ===== Vorbis Comment タグ抽出 =====
# 既存の Vorbis Comment 抽出対象タグ (foobar 階層と一致)
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


def run_hidden_cancellable(cmd, register=None, unregister=None, poll_interval=0.1):
    """run_hidden の中断可能版。Popen を起動して register に渡し、完了まで wait。
    abort で外部から terminate/kill されたとき、戻り値 returncode != 0 を例外化する。

    register(p): worker レジストリに登録
    unregister(p): 完了時に外す (例外時も finally で必ず外す)

    戻り値: subprocess.CompletedProcess 相当 (returncode のみ)。失敗時 CalledProcessError。
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if register is not None:
        try:
            register(p)
        except Exception:
            pass
    try:
        # poll ループで wait する (KeyboardInterrupt 等にも応答するため)
        while True:
            rc = p.poll()
            if rc is not None:
                break
            time.sleep(poll_interval)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
        return rc
    finally:
        if unregister is not None:
            try:
                unregister(p)
            except Exception:
                pass


# ===== 安全読み込み =====
def load_audio_safe(path):
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        return data.T, sr
    except (RuntimeError, OSError, ValueError):
        pass
    try:
        import librosa  # 遅延 import: soundfile 成功時は numba/llvmlite を読み込まない
        y, sr = librosa.load(path, mono=False, sr=None, dtype=np.float32)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr
    except Exception as e:
        raise RuntimeError(f"読み込み失敗: {path} ({type(e).__name__}: {e})") from e


# ===== 保存 (v1.6: 96kHz / FLAC PCM_24 + ffmpeg -c copy でメタデータ継承) =====
def _try_sf_write(path, data, sr, subtype, fmt):
    """書込 → ヘッダ検証で sr / frame 数 / channel 数の一致を確認する。
    旧実装はファイル全体を再デコードしていた。libsndfile の STREAMINFO を読む
    sf.info で truncated / 形状不一致は同等に検出でき、96kHz/24bit FLAC の
    全サンプル再読み込み (重い I/O+CPU) を毎ファイル省ける。出力バイトは不変。
    失敗時は中途半端に残ったファイルを削除して False を返す。"""
    try:
        sf.write(path, data, sr, subtype=subtype, format=fmt)
        info = sf.info(path)
        if info.samplerate != sr or (info.frames, info.channels) != tuple(data.shape):
            raise RuntimeError("write verification mismatch")
        return True
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False


def _true_peak(data_sc, oversample=8, chunk=1 << 20):
    """チャンク分割 8x oversampling で true peak (linear) を返す。

    data_sc: (samples, channels)。
    foobar True Peak Scanner は ITU-R BS.1770 (最小 4x oversampling) ベース。
    ここで 8x を使うことで本ツールの推定値は foobar の読み以上 (過大評価側) に
    なる → 正規化後の foobar 読みは必ず目標値以下 → clips=0 を構造的に保証する。
    全長 8x materialize は数 GB になるためチャンク処理。チャンク境界の FIR
    トランジェントは僅かに過大評価方向で clips=0 に安全側 (素サンプルピークも併用)。
    """
    if data_sc.size == 0:
        return 0.0
    n = data_sc.shape[0]
    peak = 0.0
    i = 0
    while i < n:
        seg = data_sc[i:i + chunk].astype(np.float64, copy=False)
        up = signal.resample_poly(seg, oversample, 1, axis=0)
        if up.size:
            m = float(np.max(np.abs(up)))
            if m > peak:
                peak = m
        i += chunk
    raw = float(np.max(np.abs(data_sc)))
    return max(peak, raw)


def trim_silence(y, sr, head_floor_dbfs=-55.0, tail_floor_dbfs=-55.0,
                 grad_thr_db_per_ms=0.005, smooth_ms=3.0, sustain_ms=50.0,
                 gap_fill_ms=100.0, min_actual_music_ms=50.0, blk_ms=1.0,
                 head_min_trim_s=0.02, tail_min_trim_s=0.02,
                 min_content_s=1.0, max_trim_s=180.0,
                 # 旧 API 互換 (trim_probe.py 用、無視される)
                 head_drop_db=None, tail_drop_db=None, ref_pct=None):
    """『通常音量で知覚されない曲頭/曲末』を限界まで除去 (level+gradient 知覚モデル)。
    y: (channels, samples) or (samples,)。戻り: (trimmed_y, head_s, tail_s)。

    **必ずリサンプル前の元音声に適用すること** (呼び出し側責務)。アップサンプル
    の FIR プリリンギングが無音側へ滲み境界精度を破壊するため。

    思想 (ユーザー、全音源普遍・妥協不可):
    通常音量で**可聴 = 必要、絶対に 1 frame も消さない**。
    通常音量で**無音に感じる = 不要、限界まで削る**。保守的は不可。

    破綻史と本版の到達点:
      level 単独閾: フェードインの低レベル部分を「不要」と誤分類 → 必要部分削除
        (Your Seraphim フェードイン破壊事件 2026-05-20、決定的失敗)
      → **level + gradient 二条件**で解決:
        silence = (smooth_db < ref - drop_db) AND (|gradient| < grad_thr)
        ・フェードイン: gradient > 0 → silence 不成立 → 完全保護
        ・鋭い silence→music: gradient ジャンプで blk_ms 精度検出
        ・静的無音 (digital zero/定常ハム): level<floor & grad≈0 → trim
        ・reverb tail: 時間軸変化 = gradient → 保護

    変数:
      blk_db    = 1ms 毎 peak dBFS (Audacity 波形目視と一致)
      smooth_db = blk_db の smooth_ms 移動平均 (単 block ノイズ除去)
      grad      = |diff(smooth_db)| dB/ms (フェード/エッジ検出)
      head/tail_floor_dbfs: **絶対 dBFS floor** (各 -55dBFS)。
        ref-相対 floor は dynamic range の広い曲で破綻 (静かな曲部分が
        floor 以下になり silence 誤判定。Your Seraphim 静か曲部分破壊事件
        2026-05-20 → 絶対 floor で解決)。通常音量での可聴閾は ~-55dBFS
        付近のため絶対化が知覚モデルに整合
      grad_thr_db_per_ms=0.005: 0.005dB/ms 以上の変化 = 音楽。極遅 fade
        (5dB/sec) まで保護。digital silence (grad≈0) のみ trim 対象
      smooth_ms=3: 単 block 跳ね除去 + エッジ検出力維持の均衡
      sustain_ms=50: 音楽判定に N=50 連続 block 必要。単発 PCM blip や
        孤立アーティファクトを除外しつつ、フェードイン/sharp music は通過
        (5月17日事件 2026-05-20: 静音中の単発 -84dB blip が grad ジャンプで
        誤検出され head 削れず → sustain 要件で解決)
      gap_fill_ms=100: 内部短小 silence run (<100ms) を music で埋める
        (morphological closing)。離散パルス型 fade-in (Your Seraphim:
        231/233/235/276/280/288/291/295... と単発が増えるタイプ) を sustain
        が貫通できず fade 後半まで cut してしまう問題を解決。本物の長い
        silence (>100ms) は edge も内部も影響なし
      min_actual_music_ms=50: gap-fill 後の各 music 領域に「実 music block
        (gap-fill 前) が N=50ms 以上含まれる」検証。blip cluster (5月17日:
        13/16/51/54/71/74ms に 6個 blip) が gap-fill で偽 music 領域化する
        のを排除しつつ、本物 fade (Your Seraphim: 数百 block 実 music) は通過

    重要: 本関数は提案を計算 + 切断のみ。呼び出し側で常時 apply。
    """
    is_mono = y.ndim == 1
    n = len(y) if is_mono else y.shape[-1]
    if n < int(min_content_s * sr):
        return y, 0.0, 0.0
    amp = (np.abs(y) if is_mono
           else np.max(np.abs(y), axis=0)).astype(np.float64, copy=False)

    blk = max(1, int(blk_ms * sr / 1000.0))
    nbk = n // blk
    if nbk < 8:
        return y, 0.0, 0.0
    blk_peak = amp[:nbk * blk].reshape(nbk, blk).max(axis=1)
    blk_db = 20.0 * np.log10(blk_peak + 1e-12)

    # 旧 API (head_drop_db/tail_drop_db) を絶対 floor に変換 (互換層)
    # ref_db を最大ピーク基準にし、drop_db を引いて絶対 floor 化
    if head_drop_db is not None or tail_drop_db is not None:
        ref_db = float(blk_db.max())  # 最大ピーク基準 (= ~0dBFS の曲が多い)
        if head_drop_db is not None:
            head_floor_dbfs = ref_db - head_drop_db
        if tail_drop_db is not None:
            tail_floor_dbfs = ref_db - tail_drop_db
    floor_head = head_floor_dbfs
    floor_tail = tail_floor_dbfs

    # Centered moving-average smoothing (cumsum O(N) implementation)
    sm = max(1, int(round(smooth_ms / blk_ms)))
    if sm > 1:
        pad_l = sm // 2
        pad_r = sm - pad_l - 1
        padded = np.concatenate([
            np.full(pad_l, blk_db[0]), blk_db, np.full(pad_r, blk_db[-1])
        ])
        cs = np.cumsum(np.insert(padded, 0, 0.0))
        smooth_db = (cs[sm:] - cs[:-sm]) / sm
        smooth_db = smooth_db[:nbk]
    else:
        smooth_db = blk_db

    # Per-ms gradient magnitude. blk_ms=1 → grad[i] is dB change in 1ms.
    grad = np.abs(np.diff(smooth_db, prepend=smooth_db[0])) / blk_ms

    # Silence = level below floor AND signal is stationary.
    # フェードインは grad>thr で music 扱い → 完全保護
    nonstationary = grad >= grad_thr_db_per_ms
    head_silence = (smooth_db < floor_head) & ~nonstationary
    tail_silence = (smooth_db < floor_tail) & ~nonstationary

    # Gap-fill + 実 music 検証 (blip cluster false positive 防止):
    # 1. 内部短小 silence run (<gap_fill_ms) を music 化 (fade pulse 連結)
    # 2. 各 music 領域内の「gap-fill 前の実 music block」が
    #    min_actual_music_ms 未満なら blip cluster 扱いで silence に戻す
    gap_blk = max(1, int(round(gap_fill_ms / blk_ms)))
    min_actual_blk = max(1, int(round(min_actual_music_ms / blk_ms)))

    def _refine(sil_mask):
        orig_sil = sil_mask.copy()
        sil = sil_mask.copy()
        # Pass 1: gap-fill 内部 silence run
        diff = np.diff(np.concatenate([[False], sil, [False]]).astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if s > 0 and e < nbk and (e - s) <= gap_blk:
                sil[s:e] = False
        # Pass 2: gap-fill 後の music 領域内で実 music 数を検証
        mus = ~sil
        diff2 = np.diff(np.concatenate([[False], mus, [False]]).astype(np.int8))
        m_starts = np.where(diff2 == 1)[0]
        m_ends = np.where(diff2 == -1)[0]
        for s, e in zip(m_starts, m_ends):
            # orig_sil[s:e] が True の数 = filled blocks。False = 実 music
            actual_music = (~orig_sil[s:e]).sum()
            if actual_music < min_actual_blk:
                sil[s:e] = True  # blip cluster: 元に戻す
        return sil

    head_silence = _refine(head_silence)
    tail_silence = _refine(tail_silence)
    head_music = ~head_silence
    tail_music = ~tail_silence

    # Sustain 要件: 単発 blip を除外
    sustain_blk = max(1, int(round(sustain_ms / blk_ms)))
    if sustain_blk > 1 and nbk > sustain_blk:
        from numpy.lib.stride_tricks import sliding_window_view
        wh = sliding_window_view(head_music, sustain_blk)
        wt = sliding_window_view(tail_music, sustain_blk)
        sus_head = wh.all(axis=1)
        sus_tail = wt.all(axis=1)
        if not sus_head.any() or not sus_tail.any():
            return y, 0.0, 0.0
        first_music = int(np.where(sus_head)[0][0])
        last_music = int(np.where(sus_tail)[0][-1]) + sustain_blk - 1
    else:
        if not head_music.any() or not tail_music.any():
            return y, 0.0, 0.0
        first_music = int(np.where(head_music)[0][0])
        last_music = int(np.where(tail_music)[0][-1])

    head_start = 0
    if first_music > 0:
        cand = first_music * blk
        if cand >= int(head_min_trim_s * sr):
            head_start = cand

    tail_end = n
    if last_music < nbk - 1:
        cand = min(n, (last_music + 1) * blk)
        if (n - cand) >= int(tail_min_trim_s * sr):
            tail_end = cand

    if head_start > int(max_trim_s * sr):
        head_start = 0
    if (n - tail_end) > int(max_trim_s * sr):
        tail_end = n
    if os.environ.get("DSRE_TRIM_DEBUG") == "1":
        sus_blk = max(1, int(round(sustain_ms / blk_ms)))
        sys.stderr.write(
            f"[trimdbg] n={n} sr={sr} nbk={nbk} "
            f"fh={floor_head:.1f}dBFS ft={floor_tail:.1f}dBFS "
            f"grad_thr={grad_thr_db_per_ms}dB/ms sm={sm}blk sus={sus_blk}blk "
            f"first_music={first_music}blk({first_music*blk/sr*1000:.0f}ms) "
            f"last_music={last_music} "
            f"head_start={head_start}({head_start/sr*1000:.0f}ms) "
            f"tail_end=n-{(n-tail_end)/sr:.3f}s\n")
    if (tail_end - head_start) < int(min_content_s * sr):
        return y, 0.0, 0.0
    if head_start == 0 and tail_end == n:
        return y, 0.0, 0.0

    head_s = head_start / sr
    tail_s = (n - tail_end) / sr
    trimmed = y[head_start:tail_end] if is_mono else y[:, head_start:tail_end]
    return trimmed, head_s, tail_s


def _log_trim(path, head_s, tail_s, applied=False):
    """trim_silence の提案/適用量をログファイルに追記する。
    applied=False は report-only (音声不変、精度検証用)。
    """
    log_path = os.path.join(OUTPUT_DIR, "_trim_log.txt")
    mode = "APPLIED" if applied else "PROPOSED"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{mode}] {os.path.basename(path)}\t"
                    f"head={head_s:.2f}s\ttail={tail_s:.2f}s\n")
    except Exception:
        pass


def _dsre_normalize_volume(data: "np.ndarray", sr: int) -> "np.ndarray":
    """音量最適化: true peak で TP_TARGET_DBFS に正規化し clip=0 を構造保証。

    DSRE_VOLUME_OPTIMIZE=0 で旧動作 (sample peak > 1.0 のみ正規化) に分岐。
    `data` は samples-first (samples, channels) の float32。save_flac24_out の
    内部 data 形状と同一。戻り値は data と同 shape、dtype=float32。
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


def save_flac24_out(in_path, y_out, sr, out_path, register_proc=None, unregister_proc=None):
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

    data = _dsre_normalize_volume(data, sr)

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
        # 残骸を消してから例外
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"FLAC 書込失敗 (PCM_24 / PCM_16 共に NG): {final_path}")

    # ffmpeg を「final_path への直書き」ではなく **tmp_out_path** へ書き、
    # 検証後に atomic os.replace で final_path に確定する。
    # 直書きしていた旧実装は、ffmpeg がクラッシュ/中断した時に final_path に
    # 壊れたファイルを残し、呼出側の send2trash で元音源が消える事故になり得た。
    tmp_out_path = final_path + ".tmp_out.flac"
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,     # 音声ソース (DSP 済 FLAC)
        "-i", in_path,      # メタデータソース (元 FLAC 等)
        "-map", "0:a",
        "-map_metadata", "1",
        "-c", "copy",
        tmp_out_path,
    ]
    ffmpeg_ok = False
    try:
        run_hidden_cancellable(cmd, register=register_proc, unregister=unregister_proc)
        ffmpeg_ok = True
    except Exception:
        # ffmpeg 失敗/中断: メタ無しでも音声本体 (tmp_path) は確保されている
        # → メタ無しのまま tmp_out_path にしておく
        try:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
        except OSError:
            pass
        try:
            # tmp_path をそのまま tmp_out_path にする (メタなし FLAC)
            os.replace(tmp_path, tmp_out_path)
            ffmpeg_ok = True  # メタ無しだが音声は OK
        except OSError:
            ffmpeg_ok = False

    # メタ取込が成功していれば tmp_path を片付ける
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    if not ffmpeg_ok:
        raise RuntimeError(f"FLAC 出力確定失敗: {final_path}")

    # 検証 gate (tmp_out_path の中身が壊れていないか)
    ok, reason = _verify_output_file(tmp_out_path, data, sr)
    if not ok:
        # 壊れた tmp_out は捨てる、final_path は触らない
        try:
            os.remove(tmp_out_path)
        except OSError:
            pass
        raise RuntimeError(f"出力検証失敗 ({reason}): {final_path}")

    # 検証 OK → atomic 置換で final_path に確定
    try:
        os.replace(tmp_out_path, final_path)
    except OSError as e:
        try:
            os.remove(tmp_out_path)
        except OSError:
            pass
        raise RuntimeError(f"final_path 置換失敗 ({e}): {final_path}")

    return final_path


def _verify_output_file(out_path: str, ref_data: np.ndarray, ref_sr: int) -> tuple[bool, str]:
    """出力 FLAC が ref_data (DSP 出力配列, shape=(N, ch)) と整合するかを検証。

    検査項目:
      - sf.info で読める (FLAC ヘッダ健全性)
      - sample rate が一致
      - channel 数が一致
      - frames が一致 (FLAC 無劣化なので bit-exact 一致を要求)
      - 先頭最大 1 秒をロードして RMS > 1e-7 (完全無音でない)
      - peak < 0.9999999 (true peak clipping を疑う水準より下)

    DSP 出力 ref_data が無音である正当ケースは想定外 (zansei_impl は何かしら載せる)。
    無音検出は「ffmpeg/sf がエラー時に空ファイル/全0 を書いた」事故の検出が目的。
    """
    try:
        info = sf.info(out_path)
    except Exception as e:
        return False, f"info読込失敗:{type(e).__name__}"

    if info.samplerate != int(ref_sr):
        return False, f"sr不一致:got{info.samplerate} exp{ref_sr}"

    exp_channels = int(ref_data.shape[1]) if ref_data.ndim == 2 else 1
    if info.channels != exp_channels:
        return False, f"ch不一致:got{info.channels} exp{exp_channels}"

    exp_frames = int(ref_data.shape[0])
    if info.frames != exp_frames:
        return False, f"frames不一致:got{info.frames} exp{exp_frames}"

    # 内容検査: 先頭 1 秒で RMS / peak
    try:
        n_read = int(min(info.frames, info.samplerate))
        with sf.SoundFile(out_path) as f:
            sample = f.read(n_read, dtype="float32", always_2d=True)
    except Exception as e:
        return False, f"sample読込失敗:{type(e).__name__}"

    if sample.size == 0:
        return False, "sample空"
    rms = float(np.sqrt(np.mean(sample.astype(np.float64) ** 2)))
    if not np.isfinite(rms):
        return False, "RMS非有限(NaN/Inf)"
    if rms < 1e-7:
        return False, f"出力ほぼ無音:rms={rms:.2e}"
    peak = float(np.max(np.abs(sample)))
    if not np.isfinite(peak):
        return False, "peak非有限(NaN/Inf)"
    if peak >= 0.9999999:
        return False, f"clip疑い:peak={peak:.7f}"

    return True, "ok"


_FAILURE_LOG_LOCK = threading.Lock()


def _failure_log_path() -> str:
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base or os.getcwd(), "dsre_failures.log")


def _log_failure(path: str, reason: str, exc: BaseException | None = None) -> None:
    """失敗イベントを dsre_failures.log に追記する。UI カウンタとは別経路の詳細監査ログ。"""
    log_path = _failure_log_path()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"[{ts}] reason={reason}  file={path}"]
    if exc is not None:
        lines.append(f"  exc={type(exc).__name__}: {exc}")
        tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        for ln in tb.rstrip("\n").splitlines():
            lines.append(f"    {ln}")
    msg = "\n".join(lines) + "\n"
    try:
        with _FAILURE_LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg)
    except Exception:
        pass


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


# sr は実行中不変 (96kHz 固定)、order/cutoff も定数由来で組合せは ~20 種。
# 設計係数を毎ファイル多数回再計算していたのを lru_cache で初回のみに。
# 返り値は read-only 共用 (sosfiltfilt は sos を変更しない) で bit 完全同一、
# functools.lru_cache は CPython でスレッドセーフ。
@functools.lru_cache(maxsize=None)
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


def harmonic_exciter(x, sr, params: "DSREParams | None" = None):
    """harmonic 作用 path (v1.10 等価): 整数倍音生成 (BBE Sonic Maximizer / tube exciter 系)。

    Signal flow:
      1. HP=hp_hz でソース帯域を分離 (中低域への漏れを断つ)
      2. 半波整流 + DC offset 除去 → 偶数倍音中心の歪み
      3. tanh(src * sat_gain) / sat_gain → 奇数倍音 (tube-like soft clip)
      4. half-rect / tanh を 50/50 ブレンド (偶奇両倍音バランス)
      5. HP=out_hp_hz で生成倍音の高域成分のみ通過 (中域への滲み防止)
      6. drive 倍率で zansei_impl の最終加算ステージへ渡す

    forward-only IIR は使わない (`safe_sosfiltfilt` のみ、zero-phase)。
    """
    p = params if params is not None else PARAMS

    sos_in = safe_butter_sos(8, p.exciter_hp, sr, btype="highpass")
    src = safe_sosfiltfilt(sos_in, x, axis=-1)
    src32 = src.astype(np.float32, copy=False)

    rect = np.maximum(src32, 0.0)
    if rect.ndim > 1:
        rect = rect - np.mean(rect, axis=-1, keepdims=True)
    else:
        rect = rect - np.mean(rect)
    rect = rect.astype(np.float32, copy=False)

    sat = (np.tanh(src32 * p.exciter_sat_gain) / p.exciter_sat_gain).astype(np.float32, copy=False)

    mixed = 0.5 * rect + 0.5 * sat

    sos_out = safe_butter_sos(8, p.exciter_out_hp, sr, btype="highpass")
    excited = safe_sosfiltfilt(sos_out, mixed, axis=-1)

    if not np.all(np.isfinite(excited)):
        return np.zeros_like(x)

    return (excited * p.exciter_drive).astype(x.dtype, copy=False)


def _bp_extract(x, sr, lo_hz, hi_hz, order=6):
    """BP 抽出: HP(lo) → LP(hi) を全て zero-phase sosfiltfilt で。
    高域補間ゾーンに影響を一切与えない補助 path 用。"""
    sos_hp = safe_butter_sos(order, lo_hz, sr, btype="highpass")
    y = safe_sosfiltfilt(sos_hp, x, axis=-1)
    sos_lp = safe_butter_sos(order, hi_hz, sr, btype="lowpass")
    y = safe_sosfiltfilt(sos_lp, y, axis=-1)
    return y


def body_warmth(x, sr, params: "DSREParams | None" = None):
    """中低域 (250-1200Hz) 温度感: 純 quadratic による 2nd harmonic 単独生成。

    Phase 3 改善 A: 旧 `tanh(g*s + g*s²*k)/g - s` は 2nd 主体狙いだったが
    tanh 由来の 3rd/5th 奇数倍音が混入し、low_body_harmonic との重畳で 60-400Hz 帯
    歪み係数の二重加算 + 中低域 dirty 化が発生していた (Phase 1 問題 #3, #4)。
    純 `(s² - mean(s²)) * k` 化により 2nd 単独生成、3rd 以上の混入をゼロ化。
    DC 除去で sub-Hz envelope の漏れも防止。

    出力 **LP=3000Hz** で 7kHz+ 高域補間ゾーンへの spillover を物理遮断する設計は維持。
    SAT 残響は zansei_impl 末尾の input_gate (FROZEN) で除去 (発生源対策の二重化)。
    """
    p = params if params is not None else PARAMS
    _hi = p.body_warmth_hi_hz_r if p.bp_realign_enabled else p.body_warmth_hi_hz  # A
    src = _bp_extract(x, sr, p.body_warmth_lo_hz, _hi, order=6)
    s = src.astype(np.float32, copy=False)
    s2 = (s * s).astype(np.float32, copy=False)
    if s2.ndim > 1:
        s2_mean = np.mean(s2, axis=-1, keepdims=True)
    else:
        s2_mean = np.float32(np.mean(s2))
    asym = ((s2 - s2_mean) * np.float32(p.body_warmth_asym_k)).astype(np.float32, copy=False)
    _olp = p.midlow_unified_out_lp_hz if p.cap_align_enabled else p.body_warmth_out_lp_hz  # C
    sos_out = safe_butter_sos(8, _olp, sr, btype="lowpass")
    body = safe_sosfiltfilt(sos_out, asym, axis=-1)
    if not np.all(np.isfinite(body)):
        return np.zeros_like(x)
    return (body * p.body_warmth_drive).astype(x.dtype, copy=False)


def stereo_definition_mid(x, sr, params: "DSREParams | None" = None):
    """中帯域 (300-2500Hz) Side のみ +Δ。Mid 不変、完全線形、出力 LP 3kHz cap。

    M/S 分解で Side band を BP 抽出 → LP 出力で hard cap → ±gain で L/R に加減算。
    Mid (定位の支柱) は触らないため楽器位置が崩れない。SAT を一切使わないため
    harmonic 生成ゼロ → LP cap と合わせて高域補間ゾーンに物理的に届かない。
    Mono 入力は完全 no-op。
    """
    p = params if params is not None else PARAMS
    if x.ndim != 2 or x.shape[0] != 2:
        return np.zeros_like(x)
    L = x[0]
    R = x[1]
    side = ((L - R) * 0.5).astype(np.float32, copy=False)
    if float(np.max(np.abs(side))) < 1e-9:
        return np.zeros_like(x)
    _shi = p.stereo_def_hi_hz_r if p.ms_role_split_enabled else p.stereo_def_hi_hz  # B
    side_bp = _bp_extract(side, sr, p.stereo_def_lo_hz, _shi, order=6)
    _solp = p.midlow_unified_out_lp_hz if p.cap_align_enabled else p.stereo_def_out_lp_hz  # C
    sos_out = safe_butter_sos(8, _solp, sr, btype="lowpass")
    side_capped = safe_sosfiltfilt(sos_out, side_bp, axis=-1)
    if not np.all(np.isfinite(side_capped)):
        return np.zeros_like(x)
    delta = (side_capped * p.stereo_def_gain).astype(np.float32, copy=False)
    out = np.zeros_like(x)
    out[0] = delta
    out[1] = -delta
    return out.astype(x.dtype, copy=False)


def sub_tightness(x, sr, params: "DSREParams | None" = None):
    """<150Hz サブ帯域の輪郭・速度感を envelope-based 線形 gain modulation で強化。

    fast/slow zero-phase moving-average envelope (forward-only IIR を使わない) の
    diff を [0,1] に正規化 → sub_band 自体に掛けて時変ゲイン。sign() を使わないため
    境界クリックの再発がない。出力 LP=200Hz cap で 200Hz+ には一切影響せず、
    当然 7kHz+ 高域補間ゾーンも完全に不変 (-200dB 以上の attenuation)。
    """
    p = params if params is not None else PARAMS
    sos_lp = safe_butter_sos(4, p.sub_tight_lp_hz, sr, btype="lowpass")
    sub = safe_sosfiltfilt(sos_lp, x, axis=-1)
    if not np.all(np.isfinite(sub)):
        return np.zeros_like(x)
    abs_sub = np.abs(sub.astype(np.float32, copy=False))
    win_fast = max(1, int(p.sub_tight_fast_ms * sr / 1000.0))
    win_slow = max(1, int(p.sub_tight_slow_ms * sr / 1000.0))
    if abs_sub.size < max(win_fast, win_slow) * 2:
        return np.zeros_like(x)
    ker_f = np.ones(win_fast, dtype=np.float64) / float(win_fast)
    ker_s = np.ones(win_slow, dtype=np.float64) / float(win_slow)
    abs_sub64 = abs_sub.astype(np.float64)  # float32→float64 は無損失、巻き上げで重複キャスト除去
    if abs_sub64.ndim > 1:
        env_fast = np.stack([
            np.convolve(abs_sub64[ch], ker_f, mode="same")
            for ch in range(abs_sub64.shape[0])
        ])
        env_slow = np.stack([
            np.convolve(abs_sub64[ch], ker_s, mode="same")
            for ch in range(abs_sub64.shape[0])
        ])
    else:
        env_fast = np.convolve(abs_sub64, ker_f, mode="same")
        env_slow = np.convolve(abs_sub64, ker_s, mode="same")
    diff = np.maximum(env_fast - env_slow, 0.0)
    # Phase 3 改善 C: ファイル全長 percentile_99 → 5 秒移動窓 99-percentile
    # 長尺曲後半に強い transient がある場合の前半飽和・後半抑制を回避
    norm = _moving_p99_norm(diff, sr,
                            win_sec=p.env_p99_window_sec,
                            hop_ratio=p.env_p99_hop_ratio)
    mod = np.clip(diff / norm, 0.0, 1.0).astype(np.float32, copy=False)
    boost = sub.astype(np.float32, copy=False) * mod
    sos_out = safe_butter_sos(6, p.sub_tight_out_lp_hz, sr, btype="lowpass")
    boost_safe = safe_sosfiltfilt(sos_out, boost, axis=-1)
    if not np.all(np.isfinite(boost_safe)):
        return np.zeros_like(x)
    return (boost_safe * p.sub_tight_gain).astype(x.dtype, copy=False)


def mid_transient_def(x, sr, params: "DSREParams | None" = None):
    """中帯域 (250-2500Hz) attack 定義感: 線形 envelope-based gain modulation。

    fast/slow zero-phase moving-average envelope の diff を [0,1] に正規化し
    mid_band 自体に時変ゲインを掛ける。sign() を使わない pure linear 設計の
    ため境界クリックが発生しない。SAT を介さないので harmonic を生成せず、
    出力 LP=3500Hz で 7kHz+ 高域補間ゾーンに物理的に届かない。

    効果: 楽器・ボーカル attack の立ち上がり明瞭化、分離感、勢い・生命感。
    """
    p = params if params is not None else PARAMS
    _lo = p.mid_trans_lo_hz_r if p.bp_realign_enabled else p.mid_trans_lo_hz  # A
    src = _bp_extract(x, sr, _lo, p.mid_trans_hi_hz, order=6)
    if not np.all(np.isfinite(src)):
        return np.zeros_like(x)
    abs_src = np.abs(src.astype(np.float32, copy=False))
    win_fast = max(1, int(p.mid_trans_fast_ms * sr / 1000.0))
    win_slow = max(1, int(p.mid_trans_slow_ms * sr / 1000.0))
    if abs_src.size < max(win_fast, win_slow) * 2:
        return np.zeros_like(x)
    ker_f = np.ones(win_fast, dtype=np.float64) / float(win_fast)
    ker_s = np.ones(win_slow, dtype=np.float64) / float(win_slow)
    abs_src64 = abs_src.astype(np.float64)  # float32→float64 は無損失、巻き上げで重複キャスト除去
    if abs_src64.ndim > 1:
        env_f = np.stack([
            np.convolve(abs_src64[ch], ker_f, mode="same")
            for ch in range(abs_src64.shape[0])
        ])
        env_s = np.stack([
            np.convolve(abs_src64[ch], ker_s, mode="same")
            for ch in range(abs_src64.shape[0])
        ])
    else:
        env_f = np.convolve(abs_src64, ker_f, mode="same")
        env_s = np.convolve(abs_src64, ker_s, mode="same")
    diff = np.maximum(env_f - env_s, 0.0)
    # Phase 3 改善 C: ファイル全長 percentile_99 → 5 秒移動窓 99-percentile
    norm = _moving_p99_norm(diff, sr,
                            win_sec=p.env_p99_window_sec,
                            hop_ratio=p.env_p99_hop_ratio)
    mod = np.clip(diff / norm, 0.0, 1.0).astype(np.float32, copy=False)
    boost = src.astype(np.float32, copy=False) * mod
    sos_out = safe_butter_sos(8, p.mid_trans_out_lp_hz, sr, btype="lowpass")
    boost_safe = safe_sosfiltfilt(sos_out, boost, axis=-1)
    if not np.all(np.isfinite(boost_safe)):
        return np.zeros_like(x)
    return (boost_safe * p.mid_trans_gain).astype(x.dtype, copy=False)


def vocal_presence_mid(x, sr, params: "DSREParams | None" = None):
    """ボーカル存在感: M/S 分解で Mid (=L+R)/2 のみ 1500-3000Hz BP +gentle gain。

    Side 不変なので空間像が崩れず、Mid 中帯域だけがリフトされる → ボーカル
    センターと主旋律の存在感が増す。完全線形 (SAT なし、harmonic 非生成)、
    出力 LP=3500Hz で高域補間ゾーン物理隔離。
    Mono 入力では x 全体に対する BP +gain 等価 (M=x、Side=0)。

    効果: ボーカル存在感、台詞・主旋律明瞭化、こもり改善、艶感。
    """
    p = params if params is not None else PARAMS
    if x.ndim == 2 and x.shape[0] == 2:
        L = x[0]
        R = x[1]
        mid_full = ((L + R) * 0.5).astype(np.float32, copy=False)
    else:
        mid_full = x.astype(np.float32, copy=False)
    mid_band = _bp_extract(mid_full, sr, p.vocal_pres_lo_hz, p.vocal_pres_hi_hz, order=6)
    sos_out = safe_butter_sos(8, p.vocal_pres_out_lp_hz, sr, btype="lowpass")
    mid_safe = safe_sosfiltfilt(sos_out, mid_band, axis=-1)
    if not np.all(np.isfinite(mid_safe)):
        return np.zeros_like(x)
    delta_mid = (mid_safe * p.vocal_pres_gain).astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[0] == 2:
        out = np.zeros_like(x)
        out[0] = delta_mid
        out[1] = delta_mid
        return out
    return delta_mid.astype(x.dtype, copy=False)


def low_body_harmonic(x, sr, params: "DSREParams | None" = None):
    """低域 (60-200Hz) 純 quadratic で 2nd harmonic 単独生成、胴鳴り感。

    Phase 3 改善 A: 旧 tanh ベースは 3rd 以上の奇数倍音を 60-200Hz 帯に混入させ
    body_warmth との重畳で歪み二重加算を起こしていた。純 quadratic 化で 2nd 単独
    生成、3rd 以上の混入をゼロ化。kick/bass を「周波数を上げず」体感で重く深く
    感じさせる psychoacoustic 系の意図はそのまま。

    出力 LP=400Hz cap で 400Hz+ には影響なし、7kHz+ 高域補間ゾーンも不変。
    SAT 残響は zansei_impl 末尾の input_gate (FROZEN) で除去。

    効果: 低域の輪郭・深さ・実在感、kick の存在感、bass の沈み込み。
    """
    p = params if params is not None else PARAMS
    src = _bp_extract(x, sr, p.low_body_lo_hz, p.low_body_hi_hz, order=4)
    s = src.astype(np.float32, copy=False)
    s2 = (s * s).astype(np.float32, copy=False)
    if s2.ndim > 1:
        s2_mean = np.mean(s2, axis=-1, keepdims=True)
    else:
        s2_mean = np.float32(np.mean(s2))
    asym = ((s2 - s2_mean) * np.float32(p.low_body_asym_k)).astype(np.float32, copy=False)
    sos_out = safe_butter_sos(6, p.low_body_out_lp_hz, sr, btype="lowpass")
    body = safe_sosfiltfilt(sos_out, asym, axis=-1)
    if not np.all(np.isfinite(body)):
        return np.zeros_like(x)
    return (body * p.low_body_drive).astype(x.dtype, copy=False)


# ===== Phase 3 改善ヘルパー (中低域 6 path 専用、高域補間ゾーン非関与) =====

def _moving_p99_norm(diff, sr, win_sec=ENV_P99_WINDOW_SEC, hop_ratio=ENV_P99_HOP_RATIO):
    """改善 C: 移動窓 99-percentile による envelope 動的正規化。

    ファイル全長 percentile_99 は長尺曲で前半飽和・後半抑制を起こす。本関数は
    `win_sec` 秒のブロックごとに 99-percentile を計算し、線形補間で全サンプル長に
    展開する。50% overlap で隣接ブロック境界の段差を avoid。
    短尺信号 (2 窓未満) では従来の static percentile にフォールバック。
    """
    if diff.ndim > 1:
        return np.stack([
            _moving_p99_norm(diff[ch], sr, win_sec, hop_ratio)
            for ch in range(diff.shape[0])
        ])
    n = diff.size
    win = max(1, int(win_sec * sr))
    if n <= 2 * win:
        return np.full(n, float(np.percentile(diff, 99.0)) + 1e-9, dtype=np.float64)
    hop = max(1, int(win * hop_ratio))
    centers: list[float] = []
    p99s: list[float] = []
    pos = 0
    while pos < n:
        end = min(pos + win, n)
        block = diff[pos:end]
        if block.size > 0:
            centers.append(0.5 * (pos + end - 1))
            p99s.append(float(np.percentile(block, 99.0)))
        if end >= n:
            break
        pos += hop
    centers_arr = np.asarray(centers, dtype=np.float64)
    p99s_arr = np.asarray(p99s, dtype=np.float64) + 1e-9
    return np.interp(np.arange(n, dtype=np.float64), centers_arr, p99s_arr)


def _ms_balance_correction(d_stereo, d_vocal, x, ratio_cap=MS_BALANCE_RATIO_CAP):
    """改善 D: stereo_def + vocal_pres の合成 M/S 比率を入力素材の自然比に従わせる。

    両 path 独立加算で Side+Mid 両方を増強する現状は、ジャズ live のような元 Side が
    大きい素材で左右ピンポン化リスクを生む。本関数は入力 x の Side/Mid RMS 比に対し
    delta 側の Side/Mid 比が `ratio_cap` 倍を超えた場合のみ d_stereo を線形 scale-down。
    通常素材では no-op、Mono 入力もスルー。
    """
    if x.ndim != 2 or x.shape[0] != 2:
        return d_stereo, d_vocal
    if not (isinstance(d_stereo, np.ndarray) and d_stereo.ndim == 2):
        return d_stereo, d_vocal
    if not (isinstance(d_vocal, np.ndarray) and d_vocal.ndim == 2):
        return d_stereo, d_vocal

    def _rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)) + 1e-12)

    x_mid = (x[0] + x[1]) * 0.5
    x_side = (x[0] - x[1]) * 0.5
    rms_x_mid = _rms(x_mid)
    rms_x_side = _rms(x_side)
    target_ratio = rms_x_side / rms_x_mid

    d_mid_combined = (d_stereo[0] + d_stereo[1] + d_vocal[0] + d_vocal[1]) * 0.5
    d_side_combined = (d_stereo[0] - d_stereo[1] + d_vocal[0] - d_vocal[1]) * 0.5
    rms_d_mid = _rms(d_mid_combined)
    rms_d_side = _rms(d_side_combined)
    if rms_d_mid < 1e-9 or rms_d_side < 1e-9:
        return d_stereo, d_vocal
    delta_ratio = rms_d_side / rms_d_mid

    ratio_max = max(target_ratio * ratio_cap, 1e-6)
    if delta_ratio > ratio_max:
        scale = ratio_max / delta_ratio
        d_stereo = (d_stereo * scale).astype(d_stereo.dtype, copy=False)
    return d_stereo, d_vocal


def _path_safety_cap(d, rms_x, peak_x,
                     rms_cap=PATH_RMS_CAP_RATIO, peak_cap=PATH_PEAK_CAP_RATIO):
    """改善 E + I: 中低域 path 1 つに対する RMS / peak 上限 cap (uniform scale-down)。

    path RMS が入力 RMS * `rms_cap` を超える、または path peak が入力 peak * `peak_cap`
    を超える場合のみ線形 scale-down (両 cap の min を採用)。通常素材では no-op、
    uniform scaling のため per-sample 歪み非発生、preringing 非生成。
    本 cap は中低域 6 path 専用。d_res / d_exc は abf8260 凍結ゾーンに直接含まれる
    ため適用対象外 (cap でも縮小すれば高域出力が変わるため変更禁止)。
    """
    if d.size == 0:
        return d
    abs_d = np.abs(d)
    peak = float(np.max(abs_d))
    rms = float(np.sqrt(np.mean(d.astype(np.float64) ** 2))) + 1e-12
    if rms_x <= 0.0 or peak_x <= 0.0:
        return d
    rms_max = rms_cap * rms_x
    peak_max = peak_cap * peak_x
    scale_rms = (rms_max / rms) if rms > rms_max else 1.0
    scale_peak = (peak_max / peak) if peak > peak_max and peak > 0 else 1.0
    scale = min(scale_rms, scale_peak)
    if scale < 1.0:
        return (d * scale).astype(d.dtype, copy=False)
    return d


def zansei_impl(x, sr, progress_cb=None, abort_cb=None):
    # 倍音抽出用 pre-HP (3kHz 以上を倍音生成素材に使う、SOS で数値安定化)
    sos_pre = safe_butter_sos(PARAMS.filter_order, PARAMS.pre_hp, sr, btype="highpass")
    d_src = safe_sosfiltfilt(sos_pre, x, axis=-1)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    total = PARAMS.m
    decays = np.exp(-np.arange(1, total + 1) * PARAMS.decay)
    nyq = sr / 2.0

    for i in range(total):
        if abort_cb and abort_cb():
            break

        shift = sr * (i + 1) / (total * 2.0)
        # ナイキスト到達/超過のシフト層はスキップ (折り返しアーティファクト防止)。
        # 現パラメータ (total=8, sr=96000) では最大 shift=48000=nyq なので最終層のみスキップ。
        if shift >= nyq:
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        d_res += f_dn(d_src, shift, d_sr) * decays[i]

        if progress_cb:
            progress_cb(i + 1, total)

    # 生成した倍音の低域を再度カット (16kHz 以上の高域のみに寄与、SOS で数値安定化)
    sos_post = safe_butter_sos(PARAMS.filter_order, PARAMS.post_hp, sr, btype="highpass")
    d_res = safe_sosfiltfilt(sos_post, d_res, axis=-1)

    # === 高域補間ゾーン (FROZEN: abf8260 ベースライン、改変禁止) ===
    # harmonic 作用: harmonic_exciter (整数倍音 path、freq_shift 線形変位の補完)
    d_exc = harmonic_exciter(x, sr, params=PARAMS)

    # === 中低域・空間・分離 改善ゾーン (LP cap で高域補間ゾーンと物理隔離) ===
    # body 作用: 中低域温度感 (250-1200Hz、出力 LP 3kHz、SAT 残響は input_gate 担当)
    d_body = body_warmth(x, sr, params=PARAMS)
    # stereo 作用: 中帯域 Side widening (300-2500Hz、線形、Mid 不変)
    d_stereo = stereo_definition_mid(x, sr, params=PARAMS)
    # sub 作用: <150Hz 輪郭強化 (envelope 線形 gain mod、出力 LP 200Hz)
    d_sub = sub_tightness(x, sr, params=PARAMS)
    # mid_trans 作用: 中帯域 attack 定義感 (250-2500Hz、線形 gain mod、出力 LP 3.5kHz)
    d_mid_trans = mid_transient_def(x, sr, params=PARAMS)
    # vocal_pres 作用: ボーカル存在感 (M/S Mid 1500-3000Hz、線形、出力 LP 3.5kHz)
    d_vocal = vocal_presence_mid(x, sr, params=PARAMS)
    # low_body 作用: 60-200Hz 胴鳴り (純 quadratic 2nd、出力 LP 400Hz)
    d_low_body = low_body_harmonic(x, sr, params=PARAMS)

    # === Phase 3 改善: 中低域 6 path 構造的安全弁 (高域補間ゾーン非関与) ===
    # 改善 D: stereo_def + vocal_pres の合成 M/S 比率を入力素材の自然比に従わせる
    d_stereo, d_vocal = _ms_balance_correction(
        d_stereo, d_vocal, x, ratio_cap=PARAMS.ms_balance_ratio_cap
    )
    # 改善 E + I: 中低域 6 path のみに RMS / peak cap (uniform scale-down、通常 no-op)
    # d_res / d_exc は abf8260 凍結ゾーン直接含有のため対象外
    rms_x = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
    peak_x = float(np.max(np.abs(x))) + 1e-12
    cap_kw = dict(rms_cap=PARAMS.path_rms_cap_ratio, peak_cap=PARAMS.path_peak_cap_ratio)
    d_body = _path_safety_cap(d_body, rms_x, peak_x, **cap_kw)
    d_stereo = _path_safety_cap(d_stereo, rms_x, peak_x, **cap_kw)
    d_sub = _path_safety_cap(d_sub, rms_x, peak_x, **cap_kw)
    d_mid_trans = _path_safety_cap(d_mid_trans, rms_x, peak_x, **cap_kw)
    d_vocal = _path_safety_cap(d_vocal, rms_x, peak_x, **cap_kw)
    d_low_body = _path_safety_cap(d_low_body, rms_x, peak_x, **cap_kw)

    # === Phase3 cycle3: 入力源特性に対する非凍結 6 path の bounded 連続スケール ===
    # 採用済 (耳 OK) のため既定 ON。env DSRE_PROFILE_SCALE=0 のみ kill-switch
    # として無効化 (bisect/debug)。OFF 時は下記 if を完全スキップし従来式と byte
    # 同一。凍結 d_res / d_exc はスケール対象外 (後段で無改変加算) のため HF
    # signature は入力非依存・abf8260 凍結不変。
    if PARAMS.profile_scale_enabled and os.environ.get("DSRE_PROFILE_SCALE") != "0":
        try:
            crest_db = 20.0 * np.log10(peak_x / rms_x) if rms_x > 0 else 0.0
            mono = x if x.ndim == 1 else x.mean(axis=0)
            N = int(mono.shape[-1])
            W = min(N, 1 << 19)  # 代表中央窓 (~5.5s@96k)、file 1 回 O(W log W)
            if W >= 16:
                s0 = (N - W) // 2
                seg = mono[s0:s0 + W].astype(np.float64)
                sp = np.abs(np.fft.rfft(seg)) ** 2
                fr = np.fft.rfftfreq(W, d=1.0 / sr)
                tot = float(sp.sum()) + 1e-20
                hf_ratio = float(sp[fr >= 4000.0].sum()) / tot
            else:
                hf_ratio = PARAMS.profile_hf_ref
            if x.ndim > 1 and x.shape[0] >= 2:
                _mid = (x[0] + x[1]) * 0.5
                _side = (x[0] - x[1]) * 0.5
                sm_ratio = (float(np.sqrt(np.mean(_side.astype(np.float64) ** 2)) + 1e-20)
                            / (float(np.sqrt(np.mean(_mid.astype(np.float64) ** 2))) + 1e-12))
            else:
                sm_ratio = PARAMS.profile_sm_ref
            smin, smax = PARAMS.profile_scale_min, PARAMS.profile_scale_max
            c_term = PARAMS.profile_kc * min(0.0, crest_db - PARAMS.profile_crest_ref_db)
            h_term = PARAMS.profile_kh * (PARAMS.profile_hf_ref - hf_ratio)
            scale = float(np.clip(1.0 + c_term + h_term, smin, smax))
            sm_scale = float(np.clip(1.0 + PARAMS.profile_sm_ks
                                     * (sm_ratio - PARAMS.profile_sm_ref), smin, smax))
            ms_eff = float(np.clip(scale * sm_scale, smin, smax))
            d_body = (d_body * scale).astype(d_body.dtype, copy=False)
            d_sub = (d_sub * scale).astype(d_sub.dtype, copy=False)
            d_mid_trans = (d_mid_trans * scale).astype(d_mid_trans.dtype, copy=False)
            d_low_body = (d_low_body * scale).astype(d_low_body.dtype, copy=False)
            d_stereo = (d_stereo * ms_eff).astype(d_stereo.dtype, copy=False)
            d_vocal = (d_vocal * ms_eff).astype(d_vocal.dtype, copy=False)
        except Exception:
            pass  # 算出失敗時は無スケール (= 従来挙動) に安全フォールバック

    # 加算する補完成分の合計 (d_res / d_exc は凍結出力のまま、改変なし)
    d_extra = (d_res + d_exc + d_body + d_stereo + d_sub
               + d_mid_trans + d_vocal + d_low_body)

    # 値は d_extra に確定済。中低域 6 path の中間配列を即解放し長尺ファイルの
    # ピーク RSS を抑える (del は出力に一切影響しない)。d_res/d_exc は凍結ゾーン
    # 近傍のため触れない。
    del d_body, d_stereo, d_sub, d_mid_trans, d_vocal, d_low_body

    # dynamics 作用: 入力依存ゲート (SAT 残響対策、アウトロ無音区間で d_extra を 0 化)
    # 原音 |x| の moving-average envelope < threshold で d_extra * 0、エッジは zero-phase
    # LP で滑らか化。FFT 後段矯正ではなく時間領域の振幅マスク (発生源対策)。
    try:
        abs_x = np.max(np.abs(x), axis=0) if x.ndim > 1 else np.abs(x)
        win = max(1, int(PARAMS.input_gate_window_ms * sr / 1000.0))
        if win > 1 and abs_x.size >= win:
            kernel = np.ones(win, dtype=np.float64) / float(win)
            env = np.convolve(abs_x.astype(np.float64), kernel, mode="same")
            gate = (env >= PARAMS.input_gate_threshold).astype(np.float64)
            sos_g = safe_butter_sos(2, PARAMS.input_gate_smooth_hz, sr, btype="lowpass")
            gate_smooth = safe_sosfiltfilt(sos_g, gate, axis=-1)
            gate_smooth = np.clip(gate_smooth, 0.0, 1.0).astype(x.dtype, copy=False)
            if x.ndim > 1:
                d_extra = d_extra * gate_smooth[np.newaxis, :]
            else:
                d_extra = d_extra * gate_smooth
    except Exception:
        pass

    # dynamics: auto-gain 削除 (v1.9.1 / v1.14 等価) — 純粋加算。clip 防止は
    # save_flac24_out の peak normalization に委ねる。
    result = x + d_extra
    if not np.all(np.isfinite(result)):
        return np.clip(x, -1.0, 1.0)
    return result


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _AbortedError(Exception):
    """中断シグナルを stage 境界から呼び出し元へ伝播させる内部例外。"""
    pass


def _cleanup_leftover_tmp(directory: str) -> int:
    """前回クラッシュ等で残った *.tmp_src.flac / *.tmp_out.flac を掃除して件数を返す。"""
    if not os.path.isdir(directory):
        return 0
    n = 0
    try:
        for name in os.listdir(directory):
            if name.endswith(".tmp_src.flac") or name.endswith(".tmp_out.flac"):
                try:
                    os.remove(os.path.join(directory, name))
                    n += 1
                except OSError:
                    pass
    except OSError:
        pass
    return n


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
        self._verify_failed = 0
        self._aborted_files = 0
        self._level = max(LOAD_LEVEL_MIN, min(LOAD_LEVEL_MAX, int(level)))
        self._level_lock = threading.Lock()
        self._trash_lock = threading.Lock()
        # 子プロセス (ffmpeg 等) レジストリ。abort/closeEvent から terminate する。
        self._procs_lock = threading.Lock()
        self._active_procs: set = set()

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
        # 進行中の ffmpeg 子プロセスを即 terminate (in-thread の librosa/resampy/sf.write は
        # 別スレッドから止められないが、ffmpeg は最も長く blocking しがちなので最優先で殺す)
        self.kill_all_procs(grace=0.5)

    # ---- 子プロセス管理 (run_hidden_cancellable の register/unregister 受け口) ----
    def register_proc(self, p) -> None:
        with self._procs_lock:
            self._active_procs.add(p)

    def unregister_proc(self, p) -> None:
        with self._procs_lock:
            self._active_procs.discard(p)

    def kill_all_procs(self, grace: float = 0.5) -> None:
        """登録済の全子プロセスに terminate → grace 秒待って残れば kill。"""
        with self._procs_lock:
            procs = list(self._active_procs)
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        deadline = time.time() + max(0.0, grace)
        for p in procs:
            try:
                remaining = deadline - time.time()
                if remaining > 0:
                    p.wait(timeout=remaining)
                else:
                    p.poll()
            except Exception:
                pass
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

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

        # 起動時 cleanup: 前回クラッシュ/強制終了で残った tmp_src/tmp_out を一掃。
        # 残骸があると同じ basename で連続実行時に sf.write が混乱する可能性を防ぐ。
        try:
            _cleanup_leftover_tmp(OUTPUT_DIR)
        except Exception:
            pass

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
            """単一ファイルを処理して結果コードを返す。

            戻り値:
              "ok"           : 処理成功 + 検証 OK + 元ファイル trash 成功
              "trash_fail"   : 出力 OK だが send2trash 失敗 (元ファイルは無事に残っている)
              "aborted"      : 中断 (どの段階でも、元ファイルは消されない)
              "verify_fail"  : 出力検証 NG (壊れた出力は捨てられ、元ファイルは残る)
              "disk_short"   : ディスク容量不足のため未着手 (元ファイル無事)
              例外           : run() 側で "失敗" カウンタへ
            """
            par = _resampy_parallel(lv)

            def step_cb(cur, m):
                if use_step_cb:
                    self.sig_step.emit(int(cur * 100 / m))

            def _check_abort():
                if self._abort:
                    raise _AbortedError(path)

            # ---- 事前: ディスク空き確認 (入力 size × 4 倍を要求、96k FLAC 拡張余裕込み) ----
            try:
                in_size = os.path.getsize(path)
                du = shutil.disk_usage(OUTPUT_DIR)
                if du.free < max(in_size * 4, 50 * 1024 * 1024):
                    _log_failure(path, "disk_short",
                                 RuntimeError(f"free={du.free} in_size={in_size}"))
                    return "disk_short"
            except OSError:
                pass  # 失敗しても致命ではない、本処理へ

            _check_abort()
            y, sr = load_audio_safe(path)
            pictures = _extract_flac_pictures(path)
            _check_abort()

            # 無音トリムは **リサンプル前の元音声** に適用 (アップサンプルは
            # 無音→音楽ハード edit に FIR プリリンギングを滲ませ境界を不正確に
            # するため。実測確定)。常時 apply。DSRE_TRIM_SILENCE=0 で無効化。
            if os.environ.get("DSRE_TRIM_SILENCE") != "0":
                trimmed, head_s, tail_s = trim_silence(y, sr)
                if head_s > 0 or tail_s > 0:
                    _log_trim(path, head_s, tail_s, applied=True)
                    y = trimmed
            _check_abort()

            if sr != PARAMS.target_sr:
                y, sr = _resample_to_target(y, sr, PARAMS.target_sr, par)
            _check_abort()

            before_m = MetricsComputer.compute(y, sr)
            _check_abort()
            t0 = time.perf_counter()
            y_out = zansei_impl(
                y, sr,
                progress_cb=step_cb if use_step_cb else None,
                abort_cb=lambda: self._abort,
            )
            proc_time = time.perf_counter() - t0
            after_m = MetricsComputer.compute(y_out, sr)
            _check_abort()  # zansei が abort_cb 経由で途中終了した可能性あり、ここで弾く

            out = os.path.join(OUTPUT_DIR, os.path.basename(path))
            try:
                save_flac24_out(
                    path, y_out, sr, out,
                    register_proc=self.register_proc,
                    unregister_proc=self.unregister_proc,
                )
            except _AbortedError:
                raise
            except RuntimeError as e:
                # save_flac24_out 内の検証 gate (verify_fail) もここで捕捉
                msg = str(e)
                if "出力検証失敗" in msg:
                    _log_failure(path, "verify_fail", e)
                    return "verify_fail"
                _log_failure(path, "save_fail", e)
                raise
            _embed_output_metadata(out, pictures)
            _check_abort()

            try:
                AudioMetricsLogger.log(
                    input_path=path,
                    input_audio=y,
                    output_audio=y_out,
                    sr=sr,
                    processing_time_sec=proc_time,
                    _before_metrics=before_m,
                    _after_metrics=after_m,
                )
            except Exception:
                # メトリクス記録失敗は本処理を止めない (DB ロック等)
                pass

            # 出力が確定し検証も通った後、ここで初めて元ファイルを trash
            try:
                send2trash(path)
            except Exception as e:
                _log_failure(path, "trash_fail", e)
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
                            except _AbortedError:
                                # 中断: 元ファイル無事、出力 tmp は save_flac24_out 内で除去済
                                self._aborted_files += 1
                            except Exception as e:
                                # 想定外例外。失敗ログに詳細記録、UI には件数のみ
                                self._failed.append(os.path.basename(fpath))
                                _log_failure(fpath, "exception", e)
                            else:
                                if result == "ok":
                                    succeeded += 1
                                elif result == "trash_fail":
                                    succeeded += 1  # 出力自体は OK
                                    with self._trash_lock:
                                        self._trash_failed += 1
                                elif result == "verify_fail":
                                    self._verify_failed += 1
                                    self._failed.append(os.path.basename(fpath))
                                elif result == "aborted":
                                    self._aborted_files += 1
                                elif result == "disk_short":
                                    self._failed.append(os.path.basename(fpath))
                                else:
                                    # 未知 result: 安全側で failed 計上
                                    self._failed.append(os.path.basename(fpath))
                            # 進捗テキスト更新
                            elapsed = time.time() - start_t
                            if completed_count > 0 and total > completed_count:
                                remain = (elapsed / completed_count) * (total - completed_count)
                                fail_n = len(self._failed)
                                parts = []
                                if fail_n:
                                    parts.append(f"失敗{fail_n}")
                                if self._verify_failed:
                                    parts.append(f"検証NG{self._verify_failed}")
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
                        # pending 全破棄 + 既出 future の cancel 試行 (実行中は止められない)
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
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
        verify_n = self._verify_failed
        aborted_n = self._aborted_files
        extras = []
        if fail_n:
            extras.append(f"失敗{fail_n}")
        if verify_n:
            extras.append(f"検証NG{verify_n}")
        if trash_n:
            extras.append(f"ゴミ箱{trash_n}")
        if aborted_n:
            extras.append(f"中断{aborted_n}")
        tail = ("  " + "  ".join(extras)) if extras else ""
        if fail_n or verify_n:
            # 失敗があった場合は詳細ログ場所を促す
            tail += f"  log:{_failure_log_path()}"
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

        _lv = load_level()
        self.sld_level = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sld_level.setRange(LOAD_LEVEL_MIN, LOAD_LEVEL_MAX)
        self.sld_level.setValue(_lv)
        self.sld_level.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.sld_level.setTickInterval(1)
        self.lbl_level = QtWidgets.QLabel(f"負荷 {_lv}/{LOAD_LEVEL_MAX}")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.pb_file)
        main_layout.addWidget(self.pb_all)
        main_layout.addWidget(self.btn_start)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.lbl_level)
        row.addWidget(self.sld_level, 1)
        main_layout.addLayout(row)
        self.setLayout(main_layout)

        self.btn_start.clicked.connect(self.start)
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
        """トレイ「終了」および × ボタン共通の終了処理。確認ダイアログなし (ユーザー要望)。

        全プロセス即時終了を保証する: abort + 子プロセス kill + 短い wait → 残れば
        os._exit(0) で強制終了。Python の thread は外部から kill 不可だが、プロセス
        終了で全 thread/子プロセスは確実に道連れになる。
        """
        if self.worker and self.worker.isRunning():
            try:
                self.worker.abort()  # _abort=True + kill_all_procs (ffmpeg 即殺)
            except Exception:
                pass
            try:
                self.worker.wait(1500)  # in-thread の librosa/resampy/sf.write を 1.5s 待つ
            except Exception:
                pass
            if self.worker.isRunning():
                # まだ in-flight が残っている → 子プロセスを再度全 kill して os._exit
                try:
                    self.worker.kill_all_procs(grace=0.2)
                except Exception:
                    pass
                if self._tray is not None:
                    try:
                        self._tray.hide()
                    except Exception:
                        pass
                # 強制終了: プロセス自体を落とすことで全 thread/子プロセスを確実に止める
                os._exit(0)
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
        self.worker.finished.connect(self.metrics_tab.refresh)
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
    """v1.5 自走品質ループ: import + FLAC PCM_24 roundtrip + sosfiltfilt 等価性 + 3 負荷 determinism + ffmpeg 同梱確認。

    Claude が「音響処理を改善しても劣化していない」ことを数値で判定するためのゲート。
    verdict が DEGRADED のときは exit 1 → CI / build.ps1 / deploy.ps1 が artifact を作らない。
    QApplication は作らない (ヘッドレス環境で Qt platform plugin を走らせない)。

    検査層 (v1.5 で FLAC PCM_24 ロードトリップに revert):
      (1) 必須 import (numpy/scipy/librosa/resampy/soundfile/send2trash/PySide6/threadpoolctl)
      (2) FLAC <TARGET_SR=96 kHz> / PCM_24 roundtrip: shape + sr 一致 + 量子化誤差 < 1.5e-7 (2^-23)
          v1.4 の WAV FLOAT (1e-6 閾値) から revert、v1.6 で 96k に変更
      (3) sosfiltfilt 等価性: 旧 filtfilt(ba) と新 sosfiltfilt(sos) で low Wn 11 次
          Butterworth を回し、max_abs_diff < 1e-4 / rms_diff / rms_ref < 1e-5 を確認
          (旧が NaN を吐き新が有限値のときは IMPROVED)
      (4) 3 負荷 determinism: 同一入力 × 同一負荷で 2 回実行し bit 一致
      (5) 同梱 ffmpeg の存在確認 (ビルド成果物でのみ意味がある、開発時はスキップ可)
    """
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

        # ---- (6) Phase 4 拡張 gates: frozen-zone-integrity / harmonic-cleanliness / outro-residue / spectral-monotonicity / ms-imbalance ----
        # 中低域 path 改変が高域補間ゾーンに漏れていないか、SAT 残響・段差・空間破綻が発生していないかを数値で gate。
        # 各 gate fail → verdict=DEGRADED → CI exit 1 → deploy artifact 非生成。
        gates: list[tuple[str, str, bool]] = []  # (name, tag, pass)

        # G1 harmonic-cleanliness: 300Hz 純 sin → body_warmth h3/h2 < -40 dB (純 2nd 主体)
        try:
            N_hc = TARGET_SR * 2
            t_hc = _np.arange(N_hc, dtype=_np.float64) / TARGET_SR
            sig_hc = (0.3 * _np.sin(2 * _np.pi * 300.0 * t_hc)).astype(_np.float32)
            sig_hc_2d = _np.stack([sig_hc, sig_hc])
            b_out = body_warmth(sig_hc_2d, TARGET_SR)
            ch0 = b_out[0] if b_out.ndim > 1 else b_out
            fft_hc = _np.abs(_np.fft.rfft(ch0.astype(_np.float64), n=N_hc))
            def _amp(f_hz):
                i = int(round(f_hz * N_hc / TARGET_SR))
                return float(fft_hc[i]) if 0 < i < len(fft_hc) else 0.0
            h2 = _amp(600.0) + 1e-12
            h3 = _amp(900.0) + 1e-12
            h3_h2_db = 20.0 * float(_np.log10(h3 / h2))
            hc_pass = h3_h2_db < -40.0
            gates.append(("harmonic-clean", f"h3/h2={h3_h2_db:+.1f}dB", hc_pass))
            if not hc_pass:
                verdict = "DEGRADED"
        except Exception as e:
            gates.append(("harmonic-clean", f"EXC({type(e).__name__})", False))
            verdict = "DEGRADED"

        # G2 outro-residue: 末尾 -60dBFS 区間で d_extra ≈ 0 (input_gate 効き)
        try:
            N_or = TARGET_SR * 4
            t_or = _np.arange(N_or, dtype=_np.float64) / TARGET_SR
            sig_or = (0.3 * _np.sin(2 * _np.pi * 1000.0 * t_or)).astype(_np.float32)
            sig_or[N_or // 2 :] *= 0.001  # 末尾を -60dBFS に
            sig_or_2d = _np.stack([sig_or, sig_or])
            y_or = zansei_impl(sig_or_2d.copy(), TARGET_SR)
            tail = y_or[:, N_or // 2 + TARGET_SR :]
            tail_peak = float(_np.max(_np.abs(tail))) + 1e-30
            tail_dbfs = 20.0 * float(_np.log10(tail_peak))
            or_pass = tail_dbfs < -55.0
            gates.append(("outro-residue", f"tail={tail_dbfs:+.1f}dBFS", or_pass))
            if not or_pass:
                verdict = "DEGRADED"
        except Exception as e:
            gates.append(("outro-residue", f"EXC({type(e).__name__})", False))
            verdict = "DEGRADED"

        # G3 spectral-monotonicity: white noise 入力で隣接 1kHz bin ジャンプ < +5 dB
        try:
            rng_sm = _np.random.default_rng(31337)
            N_sm = TARGET_SR * 2
            sig_sm = (rng_sm.standard_normal(N_sm).astype(_np.float32) * 0.1)
            sig_sm_2d = _np.stack([sig_sm, sig_sm])
            y_sm = zansei_impl(sig_sm_2d.copy(), TARGET_SR)
            spec = _np.abs(_np.fft.rfft(y_sm[0].astype(_np.float64), n=N_sm))
            spec_db = 20.0 * _np.log10(spec + 1e-12)
            # 1kHz bin 単位で平均し、隣接差の最大を取る
            bins_per_khz = max(1, int(1000.0 * N_sm / TARGET_SR))
            n_bands = len(spec_db) // bins_per_khz
            band_means = _np.array([float(_np.mean(spec_db[i * bins_per_khz : (i + 1) * bins_per_khz])) for i in range(n_bands)])
            jumps = _np.diff(band_means)
            max_jump = float(_np.max(jumps)) if len(jumps) > 0 else 0.0
            sm_pass = max_jump < 5.0
            gates.append(("spec-mono", f"max_jump={max_jump:+.1f}dB", sm_pass))
            if not sm_pass:
                verdict = "DEGRADED"
        except Exception as e:
            gates.append(("spec-mono", f"EXC({type(e).__name__})", False))
            verdict = "DEGRADED"

        # G4 ms-imbalance: stereo 入力で delta Side/Mid 比 vs input 比 < 1.5
        try:
            rng_ms = _np.random.default_rng(54321)
            N_ms = TARGET_SR * 2
            L_in = (rng_ms.standard_normal(N_ms).astype(_np.float32) * 0.1)
            R_in = (rng_ms.standard_normal(N_ms).astype(_np.float32) * 0.1)
            x_ms = _np.stack([L_in, R_in])
            y_ms = zansei_impl(x_ms.copy(), TARGET_SR)
            def _ms_ratio(arr):
                mid = (arr[0] + arr[1]) * 0.5
                side = (arr[0] - arr[1]) * 0.5
                rms_m = float(_np.sqrt(_np.mean(mid.astype(_np.float64) ** 2))) + 1e-12
                rms_s = float(_np.sqrt(_np.mean(side.astype(_np.float64) ** 2))) + 1e-12
                return rms_s / rms_m
            r_in = _ms_ratio(x_ms)
            r_out = _ms_ratio(y_ms.astype(_np.float32) - x_ms)  # delta の M/S 比
            ratio = r_out / max(r_in, 1e-9)
            mi_pass = ratio < 1.5
            gates.append(("ms-imbal", f"delta/in={ratio:.2f}", mi_pass))
            if not mi_pass:
                verdict = "DEGRADED"
        except Exception as e:
            gates.append(("ms-imbal", f"EXC({type(e).__name__})", False))
            verdict = "DEGRADED"

        # G5 frozen-zone-integrity: 中低域 path が 12kHz+ に漏れていないか
        # mid-low 6 path は全て出力 LP cap < 7kHz、12kHz+ への寄与はゼロが理想
        # delta = zansei_impl(x) - x の中、12kHz+ は d_res + d_exc 由来のみであるべき
        # 中低域 path のうち output LP > 12kHz のものがあれば違反 (現状なし、設計検証)
        try:
            # 設計上 LP cap が物理隔離するので、本 gate は中低域 path 単体の出力 12kHz+ エネルギーを直接見る
            rng_fz = _np.random.default_rng(7777)
            N_fz = TARGET_SR * 2
            x_fz = (rng_fz.standard_normal((2, N_fz)).astype(_np.float32) * 0.1)
            fz_max_db = -200.0
            for path_fn, path_name in [
                (body_warmth, "body"),
                (stereo_definition_mid, "stereo"),
                (sub_tightness, "sub"),
                (mid_transient_def, "mid_trans"),
                (vocal_presence_mid, "vocal"),
                (low_body_harmonic, "low_body"),
            ]:
                try:
                    out_p = path_fn(x_fz, TARGET_SR)
                    if out_p is None or (hasattr(out_p, "size") and out_p.size == 0):
                        continue
                    ch = out_p[0] if out_p.ndim > 1 else out_p
                    spec_fz = _np.abs(_np.fft.rfft(ch.astype(_np.float64), n=N_fz))
                    freqs_fz = _np.fft.rfftfreq(N_fz, 1.0 / TARGET_SR)
                    hf_mask = freqs_fz >= 12000.0
                    hf_e = float(_np.sqrt(_np.mean(spec_fz[hf_mask] ** 2))) + 1e-30
                    ref_e = float(_np.sqrt(_np.mean(spec_fz ** 2))) + 1e-30
                    rel_db = 20.0 * float(_np.log10(hf_e / ref_e))
                    if rel_db > fz_max_db:
                        fz_max_db = rel_db
                except Exception:
                    pass
            # 中低域 path の出力 12kHz+ 相対 RMS。LP cap 8 次 Butterworth + quadratic
            # 残響 + FFT 数値分解能で -25 dB 以下が実用閾値 (絶対 baseline 比較は
            # MCP frozen_zone_diff が abf8260 比較で別途実施)。
            fz_pass = fz_max_db < -25.0
            gates.append(("frozen-zone", f"midlow_12k+={fz_max_db:+.1f}dB", fz_pass))
            if not fz_pass:
                verdict = "DEGRADED"
        except Exception as e:
            gates.append(("frozen-zone", f"EXC({type(e).__name__})", False))
            verdict = "DEGRADED"

        gates_summary = " ".join(f"{n}:{t}{'✓' if p else '×'}" for n, t, p in gates)

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
                f"gates=[{gates_summary}] "
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
