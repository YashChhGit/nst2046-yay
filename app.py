# app.py
import streamlit as st
from dotenv import load_dotenv
import json, os
from tempfile import NamedTemporaryFile

from user_nlp import generate_user_preferences
from audio_features import extract_audio_features
from local_model import generate_audio

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

st.title("SoundWaves App")

if not GOOGLE_API_KEY:
    st.error("⚠️ API key not found! Please set GOOGLE_API_KEY in your .env file or environment variables.")
    st.stop()

def merge_feature_jsons(user_json: dict, audio_json: dict):
    """Merge two JSONs, using user_json values if non-null, else from audio_json."""
    key_map = {
        "TempoBPM": "Tempo",
        "TimeSignature": "Time Signature",
        "Key": "Key",
        "ToneCategory": "Tone",
        "DynamicsCategory": "Dynamics",
        "Mood": "Mood"
    }
    merged = {}
    for key in [
        "Tempo", "Time Signature", "Key", "Tone",
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

def json_to_structured_prompt(d):
    """Convert merged features JSON → Stable Audio structured metadata line."""
    tempo = d.get("Tempo")
    time_sig = d.get("Time Signature")
    key = d.get("Key")
    genre = d.get("Genre")
    subgenre = d.get("Subgenre")
    instruments = d.get("Instruments")
    mood = d.get("Mood")
    style = d.get("Style")
    tone = d.get("Tone")
    dynamics = d.get("Dynamics")

    def fmt(val):
        if not val or str(val).strip().lower() in ["none", "null", "nan"]:
            return None
        return str(val).strip()

    genre = fmt(genre)
    subgenre = fmt(subgenre)
    instruments = fmt(instruments)
    mood = fmt(mood)
    style = fmt(style)
    tone = fmt(tone)
    dynamics = fmt(dynamics)
    key = fmt(key)
    time_sig = fmt(time_sig)

    parts = []
    if genre: parts.append(f"Genre: {genre}")
    if subgenre: parts.append(f"Subgenre: {subgenre}")
    if instruments:
        instruments_str = instruments if ',' in instruments else instruments
        parts.append(f"Instruments: {instruments_str}")

    
    if style: parts.append(f"Styles: {style}")
    if mood: parts.append(f"Moods: {mood}")

    tempo_section = []
    if tempo is not None:
        tempo_section.append(f"BPM: {int(round(tempo)) if isinstance(tempo,(int,float)) else tempo}")
    if key: tempo_section.append(f"Key: {key}")
    if time_sig: tempo_section.append(f"Time Signature: {time_sig}")
    if tempo_section: parts.append(" | ".join(tempo_section))

    return " | ".join(parts)

with st.form("my_form"):
    text = st.text_area("Enter text:", "")
    uploaded_file = st.file_uploader("Choose a file")
    submitted = st.form_submit_button("Submit")

    if submitted:
        if text:
            with st.spinner("Generating response..."):
                # 1) Gemini parsing (dict)
                user_prefs = generate_user_preferences(text)
                st.write("Response:")
                try:
                    st.json(user_prefs)
                except Exception:
                    st.write(user_prefs)

                # 2) Optional audio extraction + merge
                if uploaded_file is not None:
                    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    try:
                        st.write("Audio features from uploaded file:")
                        feats = extract_audio_features(tmp_path)
                        st.json(feats)

                        merged = merge_feature_jsons(user_prefs, feats)
                        st.write("Merged Features:")
                        st.json(merged)

                        prompt_output = json_to_structured_prompt(merged)
                        st.text_area("Generated Stable Audio Prompt:", prompt_output)
                        
                        # Generate audio from the prompt
                        if prompt_output:
                            with st.spinner("Generating audio..."):
                                try:
                                    audio_filename = generate_audio(prompt_output)
                                    st.success(f"Audio generated successfully!")
                                    # Display audio player
                                    audio_path = os.path.join("app", "static", audio_filename)
                                    st.audio(audio_path, format="audio/wav")
                                except Exception as e:
                                    st.error(f"Audio generation failed: {e}")
                    except Exception as fe:
                        st.error(f"Audio feature extraction failed: {fe}")
                else:
                    # Generate prompt and audio even without uploaded file
                    prompt_output = json_to_structured_prompt(user_prefs)
                    if prompt_output:
                        st.text_area("Generated Stable Audio Prompt:", prompt_output)
                        with st.spinner("Generating audio..."):
                            try:
                                audio_filename = generate_audio(prompt_output)
                                st.success(f"Audio generated successfully!")
                                # Display audio player
                                audio_path = os.path.join("app", "static", audio_filename)
                                st.audio(audio_path, format="audio/wav")
                            except Exception as e:
                                st.error(f"Audio generation failed: {e}")
        else:
            st.warning("Please enter some text.")