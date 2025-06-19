import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}")
    else:
        print(f"No .env file found at {env_path}")
except ImportError:
    print("dotenv not available")

import google.generativeai as genai

def test_api_key():
    """Test if API key is properly configured."""
    print("=== API Key Test ===")
    
    # Check environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    fallback_key = os.getenv("GEMINI_API_KEY_FALLBACK")
    
    print(f"GEMINI_API_KEY exists: {api_key is not None}")
    print(f"GEMINI_API_KEY_FALLBACK exists: {fallback_key is not None}")
    
    if api_key:
        print(f"Primary API key length: {len(api_key)}")
        print(f"Primary API key starts with: {api_key[:10] if len(api_key) > 10 else api_key}...")
    
    if fallback_key:
        print(f"Fallback API key length: {len(fallback_key)}")
        print(f"Fallback API key starts with: {fallback_key[:10] if len(fallback_key) > 10 else fallback_key}...")
    
    # Test genai configuration
    if api_key:
        try:
            print("\nConfiguring genai with primary key...")
            genai.configure(api_key=api_key)
            
            print("Creating GenerativeModel...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            print(f"Model created: {type(model)}")
            
            print("Testing simple generation...")
            response = model.generate_content("Say hello")
            print(f"Response type: {type(response)}")
            print(f"Response text: {response.text if hasattr(response, 'text') else 'No text attribute'}")
            
        except Exception as e:
            print(f"Error with genai: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No API key available for testing")

if __name__ == "__main__":
    test_api_key()