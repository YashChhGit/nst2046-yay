# user_nlp.py
import os, re, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_user_preferences(user_text: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing in environment")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # keep model name as in your current code
        api_key=GEMINI_API_KEY,
        model_kwargs={"response_mime_type": "application/json"}
    )

    default_prompt = """You are an information extractor. Read the user's message that follows and extract ONLY the following musical features and their values if present:

- Tempo (number in BPM)
- Time Signature (string like "4/4")
- Key (string like "A minor" or "F# major")
- Tone (short descriptive phrase, e.g., "warm/mellow")
- Dynamics (short phrase, e.g., "soft", "moderate", "loud")
- Mood (short phrase, e.g., "melancholic", "energetic")
- Genre (string, e.g., "jazz", "pop", "rock", "ambient")
- Instruments (comma-separated list of instruments, e.g., "piano, drums, guitar")
- Style (short phrase describing production/performance, e.g., "acoustic", "synth-heavy", "orchestral")

Rules:
- Output MUST be valid JSON only, no prose, no Markdown.
- Use exactly these keys and this order: 
  Tempo, Time Signature, Key, Tone, Dynamics, Mood, Genre, Instruments, Style
- If a feature is missing/unspecified, set it to null (not an empty string).
- Tempo must be a number (no units), e.g., 92 (not "92 bpm").
- Normalize Key capitalization (e.g., "F# major", "A minor").
- Normalize Time Signature as "N/D" (e.g., "3/4").

Return exactly this JSON shape:
{
  "Tempo": <number or null>,
  "Time Signature": <string or null>,
  "Key": <string or null>,
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

    # Parse JSON (strict, then fallback to first {...})
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            raise ValueError(f"Model output not JSON: {txt}")
        obj = json.loads(m.group(0))

    ordered = {
        "Tempo": obj.get("Tempo"),
        "Time Signature": obj.get("Time Signature"),
        "Key": obj.get("Key"),
        "Tone": obj.get("Tone"),
        "Dynamics": obj.get("Dynamics"),
        "Mood": obj.get("Mood"),
        "Genre": obj.get("Genre"),
        "Instruments": obj.get("Instruments"),
        "Style": obj.get("Style"),
    }
    return ordered  # return dict (the Streamlit file can json.dumps it)