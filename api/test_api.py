"""
Test script cho Interview Analysis API

Chạy: python api/test_api.py
"""

import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"


def test_health():
    """Test health check."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    print("✅ Health check passed!")


def test_async_workflow(video_path: str):
    """Test async workflow: upload → analyze → poll → results."""
    print("\n" + "="*60)
    print("TEST 2: Async Workflow")
    print("="*60)
    
    # Step 1: Upload
    print("\n📤 Step 1: Upload video...")
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/upload", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    job_id = response.json()['job_id']
    print(f"✅ Upload successful! Job ID: {job_id}")
    
    # Step 2: Start analysis
    print("\n🔄 Step 2: Start analysis...")
    response = requests.post(f"{API_URL}/api/analyze/{job_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    assert response.status_code == 200
    print("✅ Analysis started!")
    
    # Step 3: Poll status
    print("\n⏳ Step 3: Polling status...")
    max_attempts = 60  # 60 attempts × 5 seconds = 5 minutes
    attempt = 0
    
    while attempt < max_attempts:
        response = requests.get(f"{API_URL}/api/status/{job_id}")
        status = response.json()['status']
        
        print(f"  Attempt {attempt + 1}: {status}")
        
        if status == 'completed':
            print("✅ Analysis completed!")
            break
        elif status == 'failed':
            print("❌ Analysis failed!")
            return
        
        time.sleep(5)
        attempt += 1
    
    if attempt >= max_attempts:
        print("❌ Timeout waiting for analysis")
        return
    
    # Step 4: Get results
    print("\n📊 Step 4: Get results...")
    response = requests.get(f"{API_URL}/api/results/{job_id}")
    print(f"Status: {response.status_code}")
    
    result = response.json()
    print("\nResults:")
    print(f"  Filename: {result['filename']}")
    print(f"  Status: {result['status']}")
    print(f"  Rating: {result['rating']}")
    print("\nScores:")
    for key, value in result['scores'].items():
        print(f"  {key.capitalize()}: {value}/10")
    
    print("\n✅ Async workflow test passed!")


def test_sync_workflow(video_path: str):
    """Test sync workflow: upload & analyze in one request."""
    print("\n" + "="*60)
    print("TEST 3: Sync Workflow (One-shot)")
    print("="*60)
    
    print("\n📤 Uploading and analyzing...")
    print("⚠️  This may take a few minutes...")
    
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/analyze-sync", files=files)
    
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nResults:")
        print(f"  Job ID: {result['job_id']}")
        print(f"  Filename: {result['filename']}")
        print(f"  Rating: {result['rating']}")
        print("\nScores:")
        for key, value in result['scores'].items():
            print(f"  {key.capitalize()}: {value}/10")
        
        print("\n✅ Sync workflow test passed!")
    else:
        print(f"❌ Request failed: {response.text}")


def test_list_jobs():
    """Test list all jobs."""
    print("\n" + "="*60)
    print("TEST 4: List Jobs")
    print("="*60)
    
    response = requests.get(f"{API_URL}/api/jobs")
    print(f"Status: {response.status_code}")
    
    result = response.json()
    print(f"\nTotal jobs: {result['total']}")
    
    if result['jobs']:
        print("\nJobs:")
        for job in result['jobs']:
            print(f"  - {job['job_id']}: {job['status']} ({job['filename']})")
    
    print("\n✅ List jobs test passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("🧪 TESTING INTERVIEW ANALYSIS API")
    print("="*60)
    
    # Check if API is running
    try:
        requests.get(API_URL)
    except requests.exceptions.ConnectionError:
        print("\n❌ API is not running!")
        print("Please start the API first:")
        print("  python api/main.py")
        return
    
    # Test 1: Health check
    test_health()
    
    # Test 2: List jobs
    test_list_jobs()
    
    # Find a test video
    video_path = None
    
    # Check recordings folder
    recordings_dir = Path("recordings")
    if recordings_dir.exists():
        video_files = list(recordings_dir.glob("*.mp4")) + list(recordings_dir.glob("*.avi"))
        if video_files:
            video_path = str(video_files[0])
    
    if not video_path:
        print("\n⚠️  No test video found!")
        print("Please provide a video file path:")
        video_path = input("Video path: ").strip()
        
        if not Path(video_path).exists():
            print("❌ Video file not found!")
            return
    
    print(f"\n📹 Using test video: {video_path}")
    
    # Ask which test to run
    print("\nWhich test do you want to run?")
    print("  1. Async workflow (recommended)")
    print("  2. Sync workflow (faster but blocks)")
    print("  3. Both")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        test_async_workflow(video_path)
    elif choice == "2":
        test_sync_workflow(video_path)
    elif choice == "3":
        test_async_workflow(video_path)
        test_sync_workflow(video_path)
    else:
        print("Invalid choice!")
        return
    
    # Final list jobs
    test_list_jobs()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
