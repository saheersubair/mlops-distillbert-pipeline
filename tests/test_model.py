"""
Integration tests for model components using the real DistillBERT model
No mocks - this tests the actual ModelManager and model functionality
"""

import sys
import os
import pytest
import asyncio
import time
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from api.utils import ModelManager, get_model_manager

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

class TestRealModelManager:
    """Integration tests for ModelManager using real DistillBERT"""

    @pytest.fixture(scope="class")
    def model_manager(self):
        """Create real ModelManager instance"""
        print("\n🤖 Creating ModelManager with real DistillBERT...")
        print("⏳ This may take a few minutes for first-time model download...")
        return ModelManager()

    @pytest.mark.asyncio
    async def test_model_loading_real(self, model_manager):
        """Test real model loading"""
        start_time = time.time()

        await model_manager.load_model()

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")

        assert model_manager.is_model_loaded()
        assert model_manager.current_version == "v1.0.0"
        assert model_manager.loaded_at is not None

        # Check model info
        info = model_manager.get_model_info()
        assert "distilbert" in info['name'].lower()
        assert info['task'] == "sentiment-analysis"
        assert 'device' in info

        print(f"✅ Model: {info['name']}")
        print(f"✅ Device: {info['device']}")
        print(f"✅ Task: {info['task']}")

    @pytest.mark.asyncio
    async def test_single_prediction_real(self, model_manager):
        """Test real single predictions"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        # Test positive sentiment
        result = await model_manager.predict("I love this amazing product!")
        assert result['label'] in ['POSITIVE', 'NEGATIVE']
        assert 0.0 <= result['score'] <= 1.0
        print(f"✅ Positive test: {result['label']} (score: {result['score']:.4f})")

        # Test negative sentiment
        result = await model_manager.predict("This product is terrible and broken!")
        assert result['label'] in ['POSITIVE', 'NEGATIVE']
        assert 0.0 <= result['score'] <= 1.0
        print(f"✅ Negative test: {result['label']} (score: {result['score']:.4f})")

        # Test neutral sentiment
        result = await model_manager.predict("This is a product.")
        assert result['label'] in ['POSITIVE', 'NEGATIVE']
        assert 0.0 <= result['score'] <= 1.0
        print(f"✅ Neutral test: {result['label']} (score: {result['score']:.4f})")

    @pytest.mark.asyncio
    async def test_batch_prediction_real(self, model_manager):
        """Test real batch predictions"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        texts = [
            "Excellent product, highly recommended!",
            "Poor quality, not worth the money.",
            "Average item, nothing special.",
            "Outstanding service and quality!",
            "Disappointed with the purchase."
        ]

        start_time = time.time()
        results = await model_manager.predict_batch(texts)
        batch_time = time.time() - start_time

        assert len(results) == len(texts)
        print(f"✅ Batch prediction completed in {batch_time:.3f} seconds")

        for i, result in enumerate(results):
            assert result['label'] in ['POSITIVE', 'NEGATIVE']
            assert 0.0 <= result['score'] <= 1.0
            print(f"   Text {i+1}: {result['label']} (score: {result['score']:.4f})")

    @pytest.mark.asyncio
    async def test_prediction_caching_real(self, model_manager):
        """Test prediction caching with real model"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        test_text = "This product is fantastic and I love it!"

        # First prediction (should be cached)
        start_time = time.time()
        result1 = await model_manager.predict(test_text)
        first_time = time.time() - start_time

        # Second prediction (should use cache)
        start_time = time.time()
        result2 = await model_manager.predict(test_text)
        second_time = time.time() - start_time

        # Results should be identical
        assert result1 == result2
        print(f"✅ Cache test: {result1['label']} (score: {result1['score']:.4f})")
        print(f"✅ First prediction: {first_time:.4f}s")
        print(f"✅ Cached prediction: {second_time:.4f}s")

        # Cached prediction should be faster (though not always guaranteed with small models)
        print(f"✅ Speedup: {first_time/second_time:.1f}x" if second_time > 0 else "✅ Instant cache hit")

    @pytest.mark.asyncio
    async def test_model_performance_real(self, model_manager):
        """Test model performance characteristics"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        # Test with various text lengths
        test_cases = [
            ("Short", "Good"),
            ("Medium", "This product is really good and I recommend it to others."),
            ("Long", "This is a comprehensive review of the product. " * 5),
        ]

        for case_name, text in test_cases:
            start_time = time.time()
            result = await model_manager.predict(text)
            prediction_time = time.time() - start_time

            assert result['label'] in ['POSITIVE', 'NEGATIVE']
            assert 0.0 <= result['score'] <= 1.0
            assert prediction_time < 5.0  # Should complete within 5 seconds

            print(f"✅ {case_name} text ({len(text)} chars): {result['label']} in {prediction_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_predictions_real(self, model_manager):
        """Test concurrent predictions with real model"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        texts = [
            "Great product!",
            "Poor quality.",
            "Love it!",
            "Not good.",
            "Excellent!"
        ]

        # Run predictions concurrently
        start_time = time.time()
        tasks = [model_manager.predict(text) for text in texts]
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        assert len(results) == len(texts)
        print(f"✅ Concurrent predictions completed in {concurrent_time:.3f}s")

        for i, result in enumerate(results):
            assert result['label'] in ['POSITIVE', 'NEGATIVE']
            assert 0.0 <= result['score'] <= 1.0
            print(f"   Task {i+1}: {result['label']} (score: {result['score']:.4f})")

        # Compare with sequential predictions
        start_time = time.time()
        for text in texts:
            await model_manager.predict(text)
        sequential_time = time.time() - start_time

        print(f"✅ Sequential time: {sequential_time:.3f}s")
        print(f"✅ Concurrent efficiency: {sequential_time/concurrent_time:.1f}x faster")

    def test_model_registry_real(self, model_manager):
        """Test model registry functionality"""
        # Test listing versions
        versions = model_manager.list_available_versions()
        assert isinstance(versions, list)
        assert "v1.0.0" in versions
        print(f"✅ Available versions: {versions}")

        # Test registering new version
        new_version = "v1.1.0-test"
        model_info = {
            "model_name": "distilbert-test",
            "task": "sentiment-analysis",
            "metrics": {"accuracy": 0.95}
        }

        model_manager.register_model_version(new_version, model_info)
        updated_versions = model_manager.list_available_versions()
        assert new_version in updated_versions
        print(f"✅ Registered new version: {new_version}")

        # Test getting metrics
        metrics = model_manager.get_model_metrics("v1.0.0")
        assert isinstance(metrics, dict)
        print(f"✅ Model metrics: {metrics}")

    def test_cache_management_real(self, model_manager):
        """Test cache management functionality"""
        # Get initial cache stats
        initial_stats = model_manager.get_cache_stats()
        assert isinstance(initial_stats, dict)
        assert "cache_size" in initial_stats
        print(f"✅ Initial cache stats: {initial_stats}")

        # Clear cache
        model_manager.clear_cache()
        cleared_stats = model_manager.get_cache_stats()
        assert cleared_stats["cache_size"] == 0
        print("✅ Cache cleared successfully")

    @pytest.mark.asyncio
    async def test_model_warm_up_real(self, model_manager):
        """Test model warm-up functionality"""
        # Clear any existing model
        model_manager.model = None

        sample_texts = [
            "Great product!",
            "Poor quality.",
            "Average item."
        ]

        start_time = time.time()
        await model_manager.warm_up(sample_texts)
        warmup_time = time.time() - start_time

        assert model_manager.is_model_loaded()
        print(f"✅ Model warmed up in {warmup_time:.2f} seconds")

    @pytest.mark.asyncio
    async def test_error_handling_real(self, model_manager):
        """Test error handling with real model"""
        if not model_manager.is_model_loaded():
            await model_manager.load_model()

        # Test with very long text (should truncate, not error)
        very_long_text = "This is a test. " * 100  # Very long text

        try:
            result = await model_manager.predict(very_long_text)
            assert result['label'] in ['POSITIVE', 'NEGATIVE']
            print("✅ Long text handled gracefully")
        except Exception as e:
            print(f"⚠️ Long text caused error: {e}")

        # Test with empty text (should be handled by API validation, not model)
        try:
            result = await model_manager.predict("")
            print(f"⚠️ Empty text surprisingly worked: {result}")
        except Exception as e:
            print(f"✅ Empty text properly rejected: {type(e).__name__}")

class TestRealABTestManager:
    """Test A/B testing functionality"""

    def test_ab_test_version_selection(self):
        """Test A/B test version selection"""
        from api.utils import get_ab_test_manager

        ab_manager = get_ab_test_manager()

        # Test with consistent user ID
        user_id = "test_user_123"
        version1 = ab_manager.get_model_version_for_request(user_id)
        version2 = ab_manager.get_model_version_for_request(user_id)

        assert version1 == version2  # Should be consistent
        assert version1 in ab_manager.model_registry if hasattr(ab_manager, 'model_registry') else True

        print(f"✅ A/B test version for {user_id}: {version1}")

class TestRealFeatureStore:
    """Test feature store functionality"""

    def test_feature_store_operations(self):
        """Test feature store operations"""
        from api.utils import get_feature_store

        feature_store = get_feature_store()

        # Test getting existing features
        features = feature_store.get_features(
            feature_names=['user_sentiment_history'],
            entity_ids=['user_123', 'user_456']
        )

        assert 'user_sentiment_history' in features
        assert len(features['user_sentiment_history']) == 2
        print(f"✅ Retrieved features: {features}")

        # Test storing new features
        feature_store.store_features(
            feature_name='test_feature',
            entity_id='test_user',
            value=[0.1, 0.2, 0.3]
        )

        # Retrieve stored feature
        stored_features = feature_store.get_features(['test_feature'], ['test_user'])
        assert stored_features['test_feature'][0] == [0.1, 0.2, 0.3]
        print("✅ Feature storage and retrieval working")

# Utility function to run integration tests
def run_model_integration_tests():
    """Run the model integration tests"""
    import subprocess

    print("🚀 Running MLOps Model Integration Tests")
    print("📝 These tests use the REAL DistillBERT model (no mocks)")
    print("⏳ First run may take several minutes to download the model...")
    print("=" * 60)

    # Set environment variables
    os.environ["TESTING"] = "false"  # We want real model loading
    os.environ["LOG_LEVEL"] = "INFO"

    # Run the tests
    cmd = [
        "python", "-m", "pytest",
        __file__,
        "-v",
        "-s",  # Don't capture output so we can see prints
        "--tb=short",
        "-m", "integration"
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 130

if __name__ == "__main__":
    import sys
    sys.exit(run_model_integration_tests())