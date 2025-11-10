"""
Simple test script to verify audio generation works locally.
"""
from local_model import generate_audio

if __name__ == "__main__":
    # Test prompt
    test_prompt = "Genre: pop | Instruments: acoustic guitar | Dynamics: moderate energy, balanced loudness | Tone: dark, warm, bass-heavy | Styles: 2000s | Moods: calm / peaceful | BPM: 120 | Key: B major"
    
    print("Testing audio generation...")
    print(f"Prompt: {test_prompt}")
    print("\nGenerating audio (this may take a while)...")
    
    try:
        filename = generate_audio(test_prompt, duration=15)
        print(f"\n✅ Success! Audio saved to: app/static/{filename}")
        print(f"You can find it at: app/static/{filename}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

