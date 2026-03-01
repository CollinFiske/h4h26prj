"""
Mock test script to verify AI model and backend connection
Run this after starting the backend server
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✓ Health Check:", response.json())
        return True
    except Exception as e:
        print("✗ Health Check Failed:", e)
        return False


def test_predict():
    """Test the predict endpoint with sample data"""
    test_data = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "has_disability": False,
        "has_pets": True,
        "has_kids": False,
        "has_medications": False,
        "other_concerns": "Live near hillside"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print("\n✓ Predict Endpoint Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print("\n✗ Predict Endpoint Failed:", e)
        return False


def test_ai_direct():
    """Test direct AI model connection (requires OPENAI_API_KEY env var)"""
    from ai import call_ai_model
    
    test_input = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "has_pets": True,
        "other_concerns": "Near wildfire zone"
    }
    
    print("\n--- Direct AI Model Test ---")
    print("Sending to OpenAI API...\n")
    
    result = call_ai_model(test_input)
    print("✓ AI Response:")
    print(json.dumps(result, indent=2))
    
    if "error" in result:
        print("✗ Error detected:", result.get("error"))
        return False
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("BACKEND CONNECTION TEST")
    print("=" * 50)
    
    # Test 1: Health check
    health_ok = test_health()
    
    if health_ok:
        # Test 2: Predict endpoint
        predict_ok = test_predict()
        
        # Test 3: Direct AI connection
        print("\n" + "=" * 50)
        try:
            ai_ok = test_ai_direct()
        except ImportError as e:
            print(f"Skipping direct AI test: {e}")
    
    print("\n" + "=" * 50)
    print("Tests complete!")
    print("=" * 50)
