"""
Text-to-Speech Demo
Takes evacuation form data, gets AI response, and reads it aloud using ElevenLabs.

Requirements:
- OPENAI_API_KEY environment variable (for AI model)
- ELEVENLABS_API_KEY environment variable (for voice synthesis)

Usage:
    python tts_demo.py
"""

import os
import requests
from ai import call_ai_model


def read_response_aloud(response_text):
    """Use ElevenLabs API to synthesize and play the response."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key:
        print("\n✗ Error: Missing ELEVENLABS_API_KEY environment variable")
        print("Set it with: $env:ELEVENLABS_API_KEY='your-api-key' (PowerShell)")
        print("Response text (not read aloud):")
        print(response_text)
        return
    
    try:
        print("\n🔊 Reading response aloud...")
        print(f"   API Key loaded: {api_key[:10]}...{api_key[-5:]}")
        
        # Rachel voice ID
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": response_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 401:
            print("\n✗ Authentication failed: Invalid or expired API key")
            print("Check your ELEVENLABS_API_KEY environment variable")
            print("Response text (fallback):")
            print(response_text)
            return
        
        response.raise_for_status()
        
        # Save audio and play it
        audio_content = response.content
        audio_file = "evacuation_guidance.mp3"
        
        with open(audio_file, "wb") as f:
            f.write(audio_content)
        
        # Play audio using system default player
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            os.startfile(audio_file)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", audio_file])
        else:  # Linux
            subprocess.run(["xdg-open", audio_file])
        
        print(f"✓ Audio saved to {audio_file} and playing...")
        
    except Exception as e:
        print(f"\n✗ ElevenLabs Error: {str(e)}")
        print("Response text (fallback):")
        print(response_text)


def main():
    """Main flow: collect form data, get AI response, read aloud."""
    
    # Sample evacuation form data (from frontend form submission)
    user_data = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "has_disability": False,
        "has_pets": True,
        "has_kids": False,
        "has_medications": False,
        "other_concerns": "Live near hillside"
    }
    
    print("\n" + "=" * 60)
    print("FIRE EVACUATION ASSISTANT - Voice Demo")
    print("=" * 60)
    print("\nForm Data:")
    print(f"  Location: ({user_data['latitude']}, {user_data['longitude']})")
    print(f"  Disability: {user_data['has_disability']}")
    print(f"  Pets: {user_data['has_pets']}")
    print(f"  Kids: {user_data['has_kids']}")
    print(f"  Medications: {user_data['has_medications']}")
    print(f"  Other Concerns: {user_data['other_concerns']}")
    print("-" * 60)
    
    # Get AI response
    print("\n🤖 Getting AI evacuation guidance...")
    ai_response = call_ai_model(user_data)
    
    print("\nAI Response:")
    print(ai_response)
    
    # Read aloud
    read_response_aloud(ai_response)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
