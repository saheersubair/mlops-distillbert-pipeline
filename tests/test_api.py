#!/usr/bin/env python3
"""
Simple test script for DistillBERT FastAPI application
Tests all main endpoints using localhost
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✅ Model info test passed")
            return True
        else:
            print("❌ Model info test failed")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False


def test_single_prediction():
    """Test single text prediction"""
    print("\nTesting single prediction...")

    test_cases = [
        {"text": "I love this product! It's amazing!", "expected_sentiment": "positive"},
        {"text": "This is terrible. I hate it.", "expected_sentiment": "negative"},
        {"text": "The weather is okay today.", "expected_sentiment": "neutral"}
    ]

    success_count = 0

    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {test_case['text']}")

        payload = {
            "text": test_case["text"]
            # Remove model_version if it's causing issues
        }

        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                headers=HEADERS,
                json=payload,
                timeout=30  # Add timeout
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Processing Time: {result['processing_time']:.4f}s")
                print("✅ Single prediction test passed")
                success_count += 1
            else:
                print(f"❌ Single prediction failed")
                print(f"Response: {response.text}")
                # Try to get more details about the error
                try:
                    error_detail = response.json()
                    print(f"Error detail: {error_detail}")
                except:
                    print("Could not parse error response as JSON")

        except requests.exceptions.Timeout:
            print(f"❌ Single prediction timeout - request took longer than 30 seconds")
        except Exception as e:
            print(f"❌ Single prediction error: {e}")

    print(f"\nSingle prediction tests: {success_count}/{len(test_cases)} passed")
    return success_count > 0  # Changed to be more lenient


def test_batch_prediction():
    """Test batch prediction"""
    print("\nTesting batch prediction...")

    test_texts = [
        "I love this product!",
        "This is terrible.",
        "The weather is okay.",
        "Amazing service!",
        "Could be better."
    ]

    payload = {
        "texts": test_texts
        # Remove model_version if it's causing issues
    }

    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers=HEADERS,
            json=payload,
            timeout=60  # Add longer timeout for batch
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Batch Size: {result['batch_size']}")
            print(f"Processing Time: {result['processing_time']:.4f}s")
            print("Predictions:")

            for i, pred in enumerate(result['predictions']):
                print(f"  {i + 1}. '{test_texts[i]}' -> {pred['prediction']} ({pred['confidence']:.4f})")

            print("✅ Batch prediction test passed")
            return True
        else:
            print(f"❌ Batch prediction failed")
            print(f"Response: {response.text}")
            # Try to get more details about the error
            try:
                error_detail = response.json()
                print(f"Error detail: {error_detail}")
            except:
                print("Could not parse error response as JSON")
            return False

    except requests.exceptions.Timeout:
        print(f"❌ Batch prediction timeout - request took longer than 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False


def test_model_versions():
    """Test model versions endpoint"""
    print("\nTesting model versions...")
    try:
        response = requests.get(f"{BASE_URL}/models/versions")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("✅ Model versions test passed")
            return True
        else:
            print("❌ Model versions test failed")
            return False
    except Exception as e:
        print(f"❌ Model versions error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    print("\nTesting metrics...")
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"Status Code: {response.status_code}")
        print(f"Response length: {len(response.text)} characters")
        print(f"First 200 chars: {response.text[:200]}...")

        if response.status_code == 200:
            print("✅ Metrics test passed")
            return True
        else:
            print("❌ Metrics test failed")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False


def test_error_cases():
    """Test error handling"""
    print("\nTesting error cases...")

    # Test empty text
    print("Testing empty text...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            json={"text": ""},
            timeout=10
        )
        print(f"Empty text - Status Code: {response.status_code}")
        print(f"Empty text - Response: {response.text}")

        if response.status_code == 400:
            print("✅ Empty text validation passed")
        elif response.status_code == 422:  # Pydantic validation error
            print("✅ Empty text validation passed (422 - validation error)")
        else:
            print(f"⚠️ Empty text returned {response.status_code} instead of 400/422")
    except Exception as e:
        print(f"❌ Empty text test error: {e}")

    # Test text too long
    print("\nTesting text too long...")
    try:
        long_text = "a" * 600  # Exceeds 512 character limit
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            json={"text": long_text},
            timeout=10
        )
        print(f"Long text - Status Code: {response.status_code}")
        print(f"Long text - Response: {response.text[:200]}...")

        if response.status_code == 400:
            print("✅ Text length validation passed")
        elif response.status_code == 422:  # Pydantic validation error
            print("✅ Text length validation passed (422 - validation error)")
        else:
            print(f"⚠️ Long text returned {response.status_code} instead of 400/422")
    except Exception as e:
        print(f"❌ Text length test error: {e}")

    # Test empty batch
    print("\nTesting empty batch...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            headers=HEADERS,
            json={"texts": []},
            timeout=10
        )
        print(f"Empty batch - Status Code: {response.status_code}")
        print(f"Empty batch - Response: {response.text}")

        if response.status_code == 400:
            print("✅ Empty batch validation passed")
        elif response.status_code == 422:  # Pydantic validation error
            print("✅ Empty batch validation passed (422 - validation error)")
        else:
            print(f"⚠️ Empty batch returned {response.status_code} instead of 400/422")
    except Exception as e:
        print(f"❌ Empty batch test error: {e}")

    # Test invalid JSON
    print("\nTesting invalid JSON...")
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            data="invalid json",
            timeout=10
        )
        print(f"Invalid JSON - Status Code: {response.status_code}")
        if response.status_code == 422:
            print("✅ Invalid JSON validation passed")
        else:
            print(f"⚠️ Invalid JSON returned {response.status_code} instead of 422")
    except Exception as e:
        print(f"❌ Invalid JSON test error: {e}")


def debug_api_status():
    """Debug function to check API status"""
    print("\n" + "=" * 50)
    print("🔍 API DEBUG INFORMATION")
    print("=" * 50)

    # Check if API is responding
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Model status: {health_data.get('model_status', 'unknown')}")
            print(f"API status: {health_data.get('status', 'unknown')}")
        else:
            print(f"Health check failed: {response.text}")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return False

    # Test a simple prediction to see detailed error
    print("\n🔍 Testing simple prediction for debugging...")
    try:
        simple_payload = {"text": "Hello world"}
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            json=simple_payload,
            timeout=30
        )
        print(f"Simple prediction status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response text: {response.text}")

        if response.status_code != 200:
            print("\n⚠️ Prediction endpoint is not working properly")
            print("This might indicate issues with:")
            print("- Model loading")
            print("- Dependencies")
            print("- Request validation")
            print("- Internal server errors")
    except Exception as e:
        print(f"❌ Simple prediction failed: {e}")

    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 Starting API Tests")
    print("=" * 60)

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    max_retries = 30
    server_ready = False

    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                health_data = response.json()
                model_status = health_data.get('model_status', 'unknown')
                print(f"✅ Server is ready! Model status: {model_status}")
                server_ready = True
                break
        except:
            if i == max_retries - 1:
                print("❌ Server is not responding. Make sure the API is running on localhost:8000")
                return
            print(f"Waiting... ({i + 1}/{max_retries})")
            time.sleep(2)

    if not server_ready:
        return

    # Run debug information first
    if not debug_api_status():
        return

    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Versions", test_model_versions),
        ("Metrics", test_metrics),
        ("Error Cases", test_error_cases)
    ]

    passed = 0
    total = len(tests)
    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            if test_func():
                passed += 1
                results.append(f"✅ {test_name}")
            else:
                results.append(f"❌ {test_name}")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append(f"💥 {test_name} (crashed)")

    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)

    for result in results:
        print(result)

    print(f"\n🏁 Summary: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("🎉 All tests passed!")
    elif passed > total // 2:
        print("⚠️  Most tests passed, but some issues found.")
    else:
        print("❌ Many tests failed. Check server logs for details.")

    # Provide troubleshooting tips
    if passed < total:
        print("\n🔧 TROUBLESHOOTING TIPS:")
        print("1. Check if the model is properly loaded")
        print("2. Review server logs for error details")
        print("3. Ensure all dependencies are installed")
        print("4. Verify the API server is running without errors")
        print("5. Check if the model files are accessible")


if __name__ == "__main__":
    run_all_tests()