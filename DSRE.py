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
HARMONIC_LAYERS = 8
HARMONIC_DECAY = 1.25
PRE_HP_CUTOFF_HZ = 3000     # 倍音抽出前のハイパス
POST_HP_CUTOFF_HZ = 12000   # 倍音生成後のハイパス (v1.8: 16k→12k で 12-16kHz 帯域を通す)
TARGET_SR = 96000           # v1.6: 本家デフォルトに戻す。192k は intermod 副作用 + 計算 2 倍の overkill だった (DSEE HX 思想は 96k 上限)
FILTER_ORDER = 11           # バターワース次数
# v1.10: Harmonic exciter (半波整流 + tanh サチュレータ) パラメータ
# freq_shift 系倍音 (線形変位) では harmonic relationship が壊れて金属臭さが出るため、
# 整数倍音を生む独立 signal path を並列追加。BBE Sonic Maximizer / tube exciter 系の手法。
EXCITER_HP_HZ = 4000        # exciter ソース帯域 (これ以上の信号から倍音生成)
EXCITER_OUT_HP_HZ = 7000    # exciter 出力 HP (これ以下は捨てる、原音中域への滲み防止)
EXCITER_DRIVE = 0.22        # 加算ブレンド比 (0.22 = 控えめ、過剰歪回避)
EXCITER_SAT_GAIN = 1.6      # tanh 入力ゲイン (歪量制御、1.6 = soft tube-like)
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
    # v1.10: Harmonic exciter (整数倍音 + tube-like saturation)
    exciter_hp: int = EXCITER_HP_HZ
    exciter_out_hp: int = EXCITER_OUT_HP_HZ
    exciter_drive: float = EXCITER_DRIVE
    exciter_sat_gain: float = EXCITER_SAT_GAIN


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



def harmonic_exciter(x, sr, hp_hz=EXCITER_HP_HZ, out_hp_hz=EXCITER_OUT_HP_HZ,
                     drive=EXCITER_DRIVE, sat_gain=EXCITER_SAT_GAIN):
    """v1.10: 整数倍音生成型 harmonic exciter (BBE Sonic Maximizer / tube exciter 系)。

    既存 zansei_impl の freq_shift 倍音 (Hilbert + 複素乗算による single-sideband shift)
    は線形周波数変位なので、原音 4kHz+8kHz が +6kHz シフトで 10kHz+14kHz に化ける
    = harmonic relationship (整数倍関係) が破壊される → 金属臭さ・不自然な高音乗りの源。

    対して半波整流 + tanh ソフト歪は、原音 f0 から 2f0, 3f0, 4f0… を自然生成し、
    harmonic relationship を保つ。両者を独立 path で blend することで、
    freq_shift だけでは到達できない「自然な厚み + 解像感」を狙う。

    Signal flow:
      1. HP=hp_hz でソース帯域を分離 (中低域への漏れを断つ)
      2. 半波整流 (max(x,0) + DC offset 除去) → 偶数倍音中心の歪み
      3. tanh(src*sat_gain)/sat_gain → 奇数倍音 (tube-like soft clip)
      4. half-rect と tanh を 50/50 ブレンド (偶奇両倍音バランス)
      5. HP=out_hp_hz で生成倍音の高域成分のみ通過 (原音帯域への漏れ防止)
      6. drive 倍率で原音に加算する量を制御
    """
    # 1. ソース帯域抽出 (4kHz 以上)
    sos_in = safe_butter_sos(8, hp_hz, sr, btype="highpass")
    src = safe_sosfiltfilt(sos_in, x, axis=-1)

    src32 = src.astype(np.float32, copy=False)

    # 2. 半波整流 (DC offset 除去、float32 精度維持)
    rect = np.maximum(src32, 0.0)
    if rect.ndim > 1:
        rect = rect - np.mean(rect, axis=-1, keepdims=True)
    else:
        rect = rect - np.mean(rect)
    rect = rect.astype(np.float32, copy=False)

    # 3. tanh サチュレータ (奇数倍音、tube-like)
    sat = (np.tanh(src32 * sat_gain) / sat_gain).astype(np.float32, copy=False)

    # 4. 偶奇ブレンド
    mixed = 0.5 * rect + 0.5 * sat

    # 5. 出力 HP で生成倍音の高域成分のみ抽出
    sos_out = safe_butter_sos(8, out_hp_hz, sr, btype="highpass")
    excited = safe_sosfiltfilt(sos_out, mixed, axis=-1)

    if not np.all(np.isfinite(excited)):
        return np.zeros_like(x)

    return (excited * drive).astype(x.dtype, copy=False)


def zansei_impl(x, sr, progress_cb=None, abort_cb=None):
    sos_pre = safe_butter_sos(PARAMS.filter_order, PARAMS.pre_hp, sr, btype="highpass")
    d_src = safe_sosfiltfilt(sos_pre, x, axis=-1)

    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)

    n_layers = PARAMS.m
    decays = np.exp(-np.arange(1, n_layers + 1) * PARAMS.decay)
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

    # v1.10: 整数倍音 exciter を並列加算 (freq_shift 系倍音と独立 path)。
    # freq_shift は線形周波数変位で harmonic relationship を破壊するため、
    # 整数倍音生成型 exciter で「自然な厚み」を補い解像感を引き上げる。
    d_exc = harmonic_exciter(
        x, sr,
        hp_hz=PARAMS.exciter_hp,
        out_hp_hz=PARAMS.exciter_out_hp,
        drive=PARAMS.exciter_drive,
        sat_gain=PARAMS.exciter_sat_gain,
    )

    result = x + d_res + d_exc

    if not np.all(np.isfinite(result)):
        return np.clip(x, -1.0, 1.0)
    return result


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
            # v1.10: hf_ratio 30% 以上上昇で IMPROVED 昇格 (DSEE HX の "失われた高域推定" の核)
            #        単に > のみなら EQUIV 並に弱い。30% は audible 改善の最小目安
            HF_GAIN_THRESHOLD = 1.30
            if hf_in > 1e-6 and hf_out >= hf_in * HF_GAIN_THRESHOLD:
                psy_notes.append(f"IMPROVED({tag},×{hf_out/hf_in:.2f})")
                if verdict == "EQUIV":
                    verdict = "IMPROVED"
            elif hf_out > hf_in:
                psy_notes.append(f"OK({tag})")
            else:
                psy_notes.append(f"WARN(no_hf_gain {tag})")

            # v1.10: harmonic_exciter 単独 sanity (整数倍音生成 + DC offset 除去)
            exc_only = harmonic_exciter(x_psy.copy(), TARGET_SR)
            # exciter 出力は HP=7kHz 通過なので低域はほぼゼロ → 倍音帯域に集中
            exc_dc = float(_np.mean(_np.abs(_np.mean(exc_only, axis=-1))))
            exc_peak = float(_np.max(_np.abs(exc_only)))
            exc_finite = bool(_np.all(_np.isfinite(exc_only)))
            if not exc_finite:
                psy_notes.append("EXC_NaN")
                verdict = "DEGRADED"
            elif exc_dc > 1e-3:
                # 半波整流の DC offset が出力 HP で除去されているはず
                psy_notes.append(f"EXC_DC_HIGH({exc_dc:.2e})")
                verdict = "DEGRADED"
            else:
                # exciter 加算で原音より hf_ratio が確実に上昇しているかを別途確認
                hf_with_exc = measure_hf_ratio(x_psy + exc_only, TARGET_SR)
                exc_hf_gain = hf_with_exc - hf_in
                # exciter 単独で hf_ratio +0.001 以上上昇を要求 (機能していることの保証)
                if exc_hf_gain < 1e-3:
                    psy_notes.append(f"EXC_NO_HF_GAIN(hf+={exc_hf_gain:+.3f})")
                    verdict = "DEGRADED"
                else:
                    psy_notes.append(f"EXC_OK(peak={exc_peak:.3f},dc={exc_dc:.1e},hf+={exc_hf_gain:+.3f})")
                    if verdict == "EQUIV":
                        verdict = "IMPROVED"
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
