import numpy as np
import librosa, scipy
from pprint import pprint

# --- Compatibility patches for newer numpy/scipy versions ---
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "int"): np.int = int
from scipy import signal as _sig
if not hasattr(_sig, "hann") and hasattr(_sig, "windows"): _sig.hann = _sig.windows.hann

# ---------- CONFIG ----------
AUDIO_PATH = "test.wav"
HOP_LENGTH = 512
METER_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]  # common beats-per-bar to check
TOP_MOODS = 6  # how many mood options to print
# ----------------------------

# ---------- Helpers for tempo & time signature ----------
def robust_ibi_frames(beat_frames):
    """Median inter-beat interval (frames) for robustness to outliers."""
    if len(beat_frames) < 3:
        return None
    diffs = np.diff(beat_frames)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))

def estimate_time_signature(y, sr, ibi_frames, hop_length=HOP_LENGTH, meter_candidates=METER_CANDIDATES):
    """
    Heuristic time-signature estimator:
    - onset envelope -> autocorrelation
    - check ACF magnitude near r*IBI for r in meter_candidates
    - choose r with strongest ACF peak; report confidence by margin over next-best
    """
    if ibi_frames is None or ibi_frames <= 0:
        return "Unknown", None, 0.0

    # Onset envelope and its autocorrelation
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(float)
    oenv = (oenv - oenv.mean()) / (oenv.std() + 1e-9)
    acf = librosa.autocorrelate(oenv, max_size=len(oenv))
    acf = acf / (acf.max() + 1e-9)
    acf[0] = 0.0

    scores = {}
    for r in meter_candidates:
        target = int(round(r * ibi_frames))
        if target <= 1 or target >= len(acf):
            scores[r] = 0.0
            continue
        half_win = max(1, int(0.1 * target))  # +/-10% window
        lo = max(1, target - half_win)
        hi = min(len(acf) - 1, target + half_win)
        scores[r] = float(np.max(acf[lo:hi+1]))

    best_r = max(scores, key=scores.get)
    best_val = scores[best_r]
    sorted_vals = sorted(scores.values(), reverse=True)
    second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    confidence = float(max(0.0, (best_val - second_val)) / (best_val + 1e-9))

    # Map beats-per-bar to a likely time signature (simple mapping)
    if best_r == 2:
        ts = "2/4"
    elif best_r == 3:
        ts = "3/4"
    elif best_r == 4:
        ts = "4/4"
    elif best_r == 5:
        ts = "5/4"
    elif best_r == 6:
        ts = "6/8"  # common feel for 6
    elif best_r == 7:
        ts = "7/8"
    elif best_r == 8:
        ts = "8/8"
    else:
        ts = f"{best_r}/4"

    return ts, best_r, confidence

# ---------- Load & features ----------
y, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)  # tempo in BPM, beats in frames (hop=512 by default)
ibi = robust_ibi_frames(beats)
ts_str, ts_beats_per_bar, ts_conf = estimate_time_signature(y, sr, ibi, hop_length=HOP_LENGTH)

rms = librosa.feature.rms(y=y)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
zcr = librosa.feature.zero_crossing_rate(y)[0]
y_harm, y_perc = librosa.effects.hpss(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = chroma.mean(axis=1)

# Extra features for mood modelling
oenv = librosa.onset.onset_strength(y=y, sr=sr)               # rhythmic drive
oenv_mean = float(np.mean(oenv))
oenv_var  = float(np.var(oenv))
flatness = librosa.feature.spectral_flatness(y=y)[0]           # noisiness/roughness proxy
flatness_mean = float(np.mean(flatness))
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
contrast_mean = float(np.mean(contrast))
if len(beats) >= 3:
    ibi_seq = np.diff(beats)
    beat_cv = float(np.std(ibi_seq) / (np.mean(ibi_seq) + 1e-9))  # lower = steadier groove
else:
    beat_cv = 1.0

# --- Key estimation (Krumhansl-Schmuckler) ---
maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
minr = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def get_key(c):
    c = c / (np.linalg.norm(c)+1e-9)
    best=(-1e9,"")
    for i in range(12):
        smaj = np.dot(c,np.roll(maj/maj.sum(),i))
        smin = np.dot(c,np.roll(minr/minr.sum(),i))
        if smaj>best[0]: best=(smaj,f"{notes[i]} major")
        if smin>best[0]: best=(smin,f"{notes[i]} minor")
    return best[1]
key = get_key(chroma_mean)

# --- Statistical summary (kept internal; not printed) ---
summary = dict(
    SampleRate=sr,
    Duration=round(duration,2),
    TempoBPM=round(float(tempo),1),
    RMSmean=float(np.mean(rms)),
    Centroid=float(np.mean(centroid)),
    Bandwidth=float(np.mean(bandwidth)),
    Rolloff=float(np.mean(rolloff)),
    ZCR=float(np.mean(zcr)),
    HarmShare=float(np.mean(np.abs(y_harm))/(np.mean(np.abs(y))+1e-9)),
    PercShare=float(np.mean(np.abs(y_perc))/(np.mean(np.abs(y))+1e-9)),
    Key=key
)

# --- CATEGORY MAPPINGS (for readability sections) ---
def tempo_category(t):
    if t < 60: return "very slow, ambient, meditative"
    elif t < 80: return "slow, downtempo, chillout"
    elif t < 100: return "moderate, laid-back groove"
    elif t < 120: return "mid-tempo, pop-like or groove-based"
    elif t < 140: return "upbeat, danceable"
    elif t < 170: return "fast-paced, energetic"
    else: return "very fast, high-BPM, intense"

def tone_category(c):
    if c < 800: return "dark, warm, bass-heavy"
    elif c < 1600: return "balanced and natural"
    elif c < 3000: return "bright, crisp, modern"
    else: return "very bright, sharp, treble-rich"

def dynamic_category(rms_val):
    if rms_val < 0.015: return "very soft, ambient texture"
    elif rms_val < 0.03: return "gentle, relaxed intensity"
    elif rms_val < 0.06: return "moderate energy, balanced loudness"
    elif rms_val < 0.1: return "strong, assertive dynamics"
    else: return "highly compressed or loud-mastered"

# ---------- Mood modelling: valence–arousal + heuristics ----------
def _z(x, lo, hi):
    """Normalize x roughly into [0..1] using min/max; clamp."""
    if hi == lo: return 0.5
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

def _sig(x):
    return float(1.0 / (1.0 + np.exp(-x)))

def estimate_valence_arousal(feat):
    """
    Explainable mapping into valence (pleasantness) and arousal (energy).
    Tuned for broad music; adjust ranges/weights to your corpus if needed.
    """
    t = _z(feat["tempo"],        40, 180)      # BPM
    e = _z(feat["rms"],        0.005, 0.12)    # loudness
    b = _z(feat["centroid"],    400, 3500)     # brightness
    p = _z(feat["perc_share"], 0.05, 0.65)     # percussive dominance
    h = _z(feat["harm_share"], 0.40, 0.98)     # harmonic dominance
    z = _z(feat["zcr"],       0.005, 0.12)     # noisiness/edginess
    f = _z(feat["flatness"],   0.05, 0.60)     # roughness/noise
    c = _z(feat["contrast"],   10.0, 35.0)     # clarity/definition
    o_m = _z(feat["oenv_mean"], 0.1, 4.0)      # transient activity
    br = _z(1.0 - feat["beat_cv"], 0.0, 1.0)   # beat regularity
    maj = float(feat["mode_is_major"])

    # AROUSAL: tempo, energy, onsets, percussive, brightness, contrast; reduced by roughness
    arousal_raw = (
        1.15*t + 1.00*e + 0.90*o_m + 0.60*p + 0.50*b + 0.25*c
        - 0.35*f - 0.20*h
    )
    arousal = _sig(2.0*(arousal_raw - 1.5))

    # VALENCE: major mode, brightness, clarity, regular groove; reduced by roughness/noise/percussive harshness
    valence_raw = (
        0.90*maj + 0.70*b + 0.50*c + 0.25*br + 0.15*h
        - 0.60*f - 0.45*z - 0.35*p
    )
    valence = _sig(2.0*(valence_raw - 1.0))

    return valence, arousal

def mood_scores_from_VA(feat):
    """
    Produce many mood options with scores in [0..1] from valence/arousal + modifiers.
    """
    V, A = estimate_valence_arousal(feat)
    t = _z(feat["tempo"], 40, 180)
    b = _z(feat["centroid"], 400, 3500)
    e = _z(feat["rms"], 0.005, 0.12)
    p = _z(feat["perc_share"], 0.05, 0.65)
    h = _z(feat["harm_share"], 0.40, 0.98)
    z = _z(feat["zcr"], 0.005, 0.12)
    f = _z(feat["flatness"], 0.05, 0.60)
    c = _z(feat["contrast"], 10.0, 35.0)
    o_m = _z(feat["oenv_mean"], 0.1, 4.0)
    br = _z(1.0 - feat["beat_cv"], 0.0, 1.0)

    S = {}
    # High-level quadrants
    S["happy / upbeat"]        = 0.55*V + 0.45*A
    S["calm / peaceful"]       = 0.65*V + 0.35*(1.0-A) + 0.2*h - 0.1*p
    S["melancholic / wistful"] = 0.75*(1.0-V) + 0.25*(1.0-A) + 0.1*h
    S["tense / anxious"]       = 0.60*(1.0-V) + 0.40*A + 0.2*f + 0.15*z

    # Texture-informed
    S["dreamy / ambient"]      = (1.0-A)*0.6 + h*0.3 + (1.0*z)*0.1
    S["aggressive / edgy"]     = 0.55*A + 0.25*z + 0.20*f + 0.15*p
    S["energetic / pumped"]    = 0.70*A + 0.20*t + 0.10*p
    S["groovy / danceable"]    = 0.45*A + 0.35*br + 0.20*t + 0.10*o_m

    # Timbre & nuance
    S["bright / optimistic"]   = 0.60*V + 0.40*b + 0.10*c
    S["dark / mysterious"]     = 0.50*(1.0*V) + 0.50*(1.0*b) + 0.20*(1.0*c) + 0.10*f
    S["warm / romantic"]       = 0.60*V + 0.20*(1.0*b) + 0.20*h
    S["somber / reflective"]   = 0.65*(1.0-V) + 0.35*(1.0-A)

    # Rhythm-centric
    S["driving / urgent"]      = 0.55*A + 0.25*p + 0.20*o_m + 0.15*br
    S["floaty / ethereal"]     = (1.0-A)*0.5 + 0.30*h + 0.20*(1.0*c)

    # Clamp to [0..1]
    for k, v in S.items():
        S[k] = float(max(0.0, min(1.0, v)))

    S["_valence"] = float(V)
    S["_arousal"] = float(A)
    return S

# --- Build the feature dict for mood model ---
feat = {
    "tempo": float(tempo),
    "rms": float(np.mean(rms)),
    "centroid": float(np.mean(centroid)),
    "bandwidth": float(np.mean(bandwidth)),
    "rolloff": float(np.mean(rolloff)),
    "zcr": float(np.mean(zcr)),
    "harm_share": float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y)) + 1e-9)),
    "perc_share": float(np.mean(np.abs(y_perc)) / (np.mean(np.abs(y)) + 1e-9)),
    "flatness": flatness_mean,
    "contrast": contrast_mean,
    "oenv_mean": oenv_mean,
    "oenv_var": oenv_var,
    "beat_cv": beat_cv,
    "mode_is_major": 1.0 if "major" in key.lower() else 0.0,
}

# --- Interpretation ---
print("\n=== HUMAN INTERPRETATION ===")
print(f"Tempo (exact): {summary['TempoBPM']:.1f} BPM")
if ts_str != 'Unknown':
    print(f"Time Signature (estimated): {ts_str} ")
else:
    print("Time Signature (estimated): Unknown")

print(f"Key (estimated): {summary['Key']}")
print(f"Tempo category: {tempo_category(summary['TempoBPM'])}")
print(f"Tone: {tone_category(summary['Centroid'])}")
print(f"Dynamics: {dynamic_category(summary['RMSmean'])}")

# Improved mood output (multiple options with scores)
moods = mood_scores_from_VA(feat)
top = sorted([(k, v) for k, v in moods.items() if not k.startswith("_")],
             key=lambda kv: kv[1], reverse=True)[:TOP_MOODS]
print(f"Mood: {top[0][0]}")

