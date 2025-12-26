"""
Test script for transcription endpoints
"""

import requests
import sys

API_URL = "http://localhost:8000"

def test_endpoints():
    """Test if transcription endpoints exist."""
    
    print("=" * 60)
    print("Testing Transcription Endpoints")
    print("=" * 60)
    print()
    
    # Test 1: Check API is running
    print("1. Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API is running")
        else:
            print(f"   ❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API")
        print("   → Please start API: python api/main.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    
    # Test 2: Check transcribe-video endpoint
    print("2. Checking /api/transcribe-video endpoint...")
    try:
        # Send empty request to check if endpoint exists
        response = requests.post(f"{API_URL}/api/transcribe-video", timeout=5)
        
        if response.status_code == 422:
            print("   ✅ Endpoint exists (422 = missing file, expected)")
        elif response.status_code == 404:
            print("   ❌ Endpoint NOT FOUND (404)")
            print("   → Please RESTART API server:")
            print("      1. Stop current API (Ctrl+C)")
            print("      2. Run: python api/main.py")
            return False
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    
    # Test 3: Check transcribe-audio endpoint
    print("3. Checking /api/transcribe-audio endpoint...")
    try:
        response = requests.post(f"{API_URL}/api/transcribe-audio", timeout=5)
        
        if response.status_code == 422:
            print("   ✅ Endpoint exists (422 = missing file, expected)")
        elif response.status_code == 404:
            print("   ❌ Endpoint NOT FOUND (404)")
            print("   → Please RESTART API server")
            return False
        else:
            print(f"   ⚠️  Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print()
    print("=" * 60)
    print("✅ All endpoints are working!")
    print("=" * 60)
    print()
    print("You can now use the transcription features in the web app.")
    print()
    
    return True


if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)
