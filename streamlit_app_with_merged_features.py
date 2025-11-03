import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import os
import re
import numpy as np
import librosa, scipy
from tempfile import NamedTemporaryFile

# --- Compatibility patches for newer numpy/scipy versions (for librosa) ---
if not hasattr(np, "complex"): np.complex = complex  # type: ignore[attr-defined]
if not hasattr(np, "float"):   np.float   = float    # type: ignore[attr-defined]
if not hasattr(np, "int"):     np.int     = int      # type: ignore[attr-defined]
from scipy import signal as _sig
if not hasattr(_sig, "hann") and hasattr(_sig, "windows"): _sig.hann = _sig.windows.hann  # type: ignore[attr-defined]

# Load environment variables
load_dotenv()

# Get API key and verify it's loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Bomboclatt App")

# Check if API key is loaded
if not GOOGLE_API_KEY:
    st.error("⚠️ API key not found! Please set GOOGLE_API_KEY in your .env file or environment variables.")
    st.stop()

def generate_user_preferences(user_text: str):
    try:
        # Use a valid model name and force JSON output
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GOOGLE_API_KEY,
            model_kwargs={"response_mime_type": "application/json"}
        )

        default_prompt = """You are an information extractor. Read the user's message that follows and extract ONLY the following musical features and their values if present:

- Tempo (number in BPM)
- Time Signature (string like "4/4")
- Key (string like "A minor" or "F# major")
- Tempo category (one of: "very slow", "slow", "moderate", "fast", "very fast")
- Tone (short descriptive phrase, e.g., "warm/mellow")
- Dynamics (short phrase, e.g., "soft", "moderate", "loud")
- Mood (short phrase, e.g., "melancholic", "energetic")
- Genre (string, e.g., "jazz", "pop", "rock", "ambient")
- Instruments (comma-separated list of instruments, e.g., "piano, drums, guitar")
- Style (short phrase describing production/performance, e.g., "acoustic", "synth-heavy", "orchestral")

Rules:
- Output MUST be valid JSON only, no prose, no Markdown.
- Use exactly these keys and this order: 
  Tempo, Time Signature, Key, Tempo category, Tone, Dynamics, Mood, Genre, Instruments, Style
- If a feature is missing/unspecified, set it to null (not an empty string).
- Tempo must be a number (no units), e.g., 92 (not "92 bpm").
- Normalize Key capitalization (e.g., "F# major", "A minor").
- Normalize Time Signature as "N/D" (e.g., "3/4").
- Infer "Tempo category" from Tempo if only BPM is provided (≤40 very slow, 41–70 slow, 71–110 moderate, 111–160 fast, >160 very fast). If neither is given, leave both as null.

Return exactly this JSON shape:
{
  "Tempo": <number or null>,
  "Time Signature": <string or null>,
  "Key": <string or null>,
  "Tempo category": <string or null>,
  "Tone": <string or null>,
  "Dynamics": <string or null>,
  "Mood": <string or null>,
  "Genre": <string or null>,
  "Instruments": <string or null>,
  "Style": <string or null>
}

Now process the user's message:
"""

        raw = llm.invoke(default_prompt + user_text).content
        if not raw:
            raise ValueError("Model returned empty output")
        txt = str(raw).strip()

        # Strip Markdown code fences like ```json ... ```
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```$", "", txt)

        # Try strict JSON first; if that fails, extract the first {...} block
        try:
            obj = json.loads(txt)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                raise ValueError(f"Model output not JSON: {txt}")
            obj = json.loads(m.group(0))
        # Optional: ensure keys exist and in order
        ordered = {
            "Tempo": obj.get("Tempo"),
            "Time Signature": obj.get("Time Signature"),
            "Key": obj.get("Key"),
            "Tempo category": obj.get("Tempo category"),
            "Tone": obj.get("Tone"),
            "Dynamics": obj.get("Dynamics"),
            "Mood": obj.get("Mood"),
            "Genre": obj.get("Genre"),
            "Instruments": obj.get("Instruments"),
            "Style": obj.get("Style"),
        }
        return json.dumps(ordered, indent=2)
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_response(prompt):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ================== AUDIO FEATURE EXTRACTION (runs on uploaded file) ==================
HOP_LENGTH = 512
METER_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]

def robust_ibi_frames(beat_frames):
    if beat_frames is None or len(beat_frames) < 3:
        return None
    diffs = np.diff(beat_frames)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))

def estimate_time_signature(y, sr, ibi_frames, hop_length=HOP_LENGTH, meter_candidates=METER_CANDIDATES):
    if ibi_frames is None or ibi_frames <= 0:
        return "Unknown", None, 0.0
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
        half_win = max(1, int(0.1 * target))
        lo = max(1, target - half_win)
        hi = min(len(acf) - 1, target + half_win)
        scores[r] = float(np.max(acf[lo:hi+1]))
    best_r = max(scores, key=scores.get)
    best_val = scores[best_r]
    sorted_vals = sorted(scores.values(), reverse=True)
    second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    # confidence unused in UI but kept for completeness
    _confidence = float(max(0.0, (best_val - second_val)) / (best_val + 1e-9))
    if best_r == 2: ts = "2/4"
    elif best_r == 3: ts = "3/4"
    elif best_r == 4: ts = "4/4"
    elif best_r == 5: ts = "5/4"
    elif best_r == 6: ts = "6/8"
    elif best_r == 7: ts = "7/8"
    elif best_r == 8: ts = "8/8"
    else: ts = f"{best_r}/4"
    return ts, best_r, _confidence

def tempo_category(t):
    t = float(t)
    if t < 60: return "very slow, ambient, meditative"
    elif t < 80: return "slow, downtempo, chillout"
    elif t < 100: return "moderate, laid-back groove"
    elif t < 120: return "mid-tempo, pop-like or groove-based"
    elif t < 140: return "upbeat, danceable"
    elif t < 170: return "fast-paced, energetic"
    else: return "very fast, high-BPM, intense"

def tone_category(c):
    c = float(c)
    if c < 800: return "dark, warm, bass-heavy"
    elif c < 1600: return "balanced and natural"
    elif c < 3000: return "bright, crisp, modern"
    else: return "very bright, sharp, treble-rich"

def dynamic_category(rms_val):
    rms_val = float(rms_val)
    if rms_val < 0.015: return "very soft, ambient texture"
    elif rms_val < 0.03: return "gentle, relaxed intensity"
    elif rms_val < 0.06: return "moderate energy, balanced loudness"
    elif rms_val < 0.1: return "strong, assertive dynamics"
    else: return "highly compressed or loud-mastered"

_KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
_NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def _estimate_key_from_chroma(chroma_mean):
    c = chroma_mean / (np.linalg.norm(chroma_mean)+1e-9)
    best=(-1e9,"")
    for i in range(12):
        smaj = np.dot(c, np.roll(_KS_MAJ/_KS_MAJ.sum(), i))
        smin = np.dot(c, np.roll(_KS_MIN/_KS_MIN.sum(), i))
        if smaj > best[0]: best = (smaj, f"{_NOTES[i]} major")
        if smin > best[0]: best = (smin, f"{_NOTES[i]} minor")
    return best[1]

def _z(x, lo, hi):
    if hi == lo: return 0.5
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

def _sig(x):
    return float(1.0 / (1.0 + np.exp(-x)))

def _estimate_valence_arousal(feat):
    t = _z(feat["tempo"],        40, 180)
    e = _z(feat["rms"],        0.005, 0.12)
    b = _z(feat["centroid"],    400, 3500)
    p = _z(feat["perc_share"], 0.05, 0.65)
    h = _z(feat["harm_share"], 0.40, 0.98)
    z = _z(feat["zcr"],       0.005, 0.12)
    f = _z(feat["flatness"],   0.05, 0.60)
    c = _z(feat["contrast"],   10.0, 35.0)
    o_m = _z(feat["oenv_mean"], 0.1, 4.0)
    br = _z(1.0 - feat["beat_cv"], 0.0, 1.0)
    maj = float(feat["mode_is_major"])
    arousal_raw = (1.15*t + 1.00*e + 0.90*o_m + 0.60*p + 0.50*b + 0.25*c - 0.35*f - 0.20*h)
    arousal = _sig(2.0*(arousal_raw - 1.5))
    valence_raw = (0.90*maj + 0.70*b + 0.50*c + 0.25*br + 0.15*h - 0.60*f - 0.45*z - 0.35*p)
    valence = _sig(2.0*(valence_raw - 1.0))
    return valence, arousal

def mood_scores_from_VA(feat):
    """
    Produce many mood options with scores in [0..1] from valence/arousal + modifiers.
    """
    V, A = _estimate_valence_arousal(feat)
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

def extract_audio_features(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # Ensure tempo is a scalar (librosa may return a 0-d array or 1-d array)
    tempo = float(np.atleast_1d(tempo).astype(float).ravel()[0])
    ibi = robust_ibi_frames(beats)
    ts_str, _, _ = estimate_time_signature(y, sr, ibi, hop_length=HOP_LENGTH)

    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    y_harm, y_perc = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key = _estimate_key_from_chroma(chroma_mean)

    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    if len(beats) >= 3:
        ibi_seq = np.diff(beats)
        beat_cv = float(np.std(ibi_seq) / (np.mean(ibi_seq) + 1e-9))
    else:
        beat_cv = 1.0

    feat = {
        "tempo": tempo,
        "rms": float(np.mean(rms)),
        "centroid": float(np.mean(centroid)),
        "bandwidth": float(np.mean(bandwidth)),
        "rolloff": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
        "harm_share": float(np.mean(np.abs(y_harm)) / (np.mean(np.abs(y)) + 1e-9)),
        "perc_share": float(np.mean(np.abs(y_perc)) / (np.mean(np.abs(y)) + 1e-9)),
        "flatness": float(np.mean(flatness)),
        "contrast": float(np.mean(contrast)),
        "oenv_mean": float(np.mean(oenv)),
        "oenv_var": float(np.var(oenv)),
        "beat_cv": beat_cv,
        "mode_is_major": 1.0 if "major" in key.lower() else 0.0,
    }

    moods = mood_scores_from_VA(feat)
    top_label = sorted([(k, v) for k, v in moods.items() if not k.startswith("_")],
                       key=lambda kv: kv[1], reverse=True)[0][0]

    return {
        "TempoBPM": round(tempo, 1),
        "TimeSignature": ts_str,
        "Key": key,
        "TempoCategory": tempo_category(tempo),
        "ToneCategory": tone_category(float(np.mean(centroid))),
        "DynamicsCategory": dynamic_category(float(np.mean(rms))),
        "Mood": top_label,
    }

def merge_feature_jsons(user_json: dict, audio_json: dict):
    """Merge two JSONs, using user_json values if non-null, else from audio_json."""
    key_map = {
        "TempoBPM": "Tempo",
        "TimeSignature": "Time Signature",
        "Key": "Key",
        "TempoCategory": "Tempo category",
        "ToneCategory": "Tone",
        "DynamicsCategory": "Dynamics",
        "Mood": "Mood"
    }

    merged = {}
    for key in [
        "Tempo", "Time Signature", "Key", "Tempo category", "Tone",
        "Dynamics", "Mood", "Genre", "Instruments", "Style"
    ]:
        user_val = user_json.get(key)
        audio_val = None
        for a_k, u_k in key_map.items():
            if u_k == key:
                audio_val = audio_json.get(a_k)
                break
        merged[key] = user_val if user_val not in [None, "", "null", "Null", "NULL"] else audio_val
    return merged
# ================== END AUDIO FEATURE EXTRACTION ==================

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Mi bomba",
    )
    uploaded_file = st.file_uploader("Choose a file")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if text:
            with st.spinner("Generating response..."):
                response = generate_user_preferences(text)
                st.write("Response:")
                try:
                    st.json(json.loads(response))
                except Exception:
                    st.write(response)  # fallback if it's an error string
                # --- If a file was uploaded, save it and run feature extraction ---
                if uploaded_file is not None:
                    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        st.write("Audio features from uploaded file:")
                        feats = extract_audio_features(tmp_path)
                        st.json(feats)
                        try:
                            user_dict = json.loads(response)
                            merged = merge_feature_jsons(user_dict, feats)
                            st.write("Merged Features:")
                            st.json(merged)
                        except Exception as me:
                            st.error(f"Failed to merge JSONs: {me}")
                    except Exception as fe:
                        st.error(f"Audio feature extraction failed: {fe}")
        else:
            st.warning("Please enter some text.")