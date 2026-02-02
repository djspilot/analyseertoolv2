"""
Unit tests for DataService
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.services.data_service import DataService
from src.api.cache import ENABLE_CACHE


@pytest.mark.unit
class TestDataService:
    """Test DataService class."""

    @pytest.fixture
    def data_service(self):
        """Create a DataService instance with caching disabled for testing."""
        return DataService(cache_enabled=False)

    @pytest.fixture
    def mock_database(self):
        """Mock database functions."""
        with patch("src.api.database.init_db") as mock_init, \
             patch("src.api.database.get_all_activities") as mock_get, \
             patch("src.api.database.get_activity_count") as mock_count:
            yield {
                "init": mock_init,
                "get": mock_get,
                "count": mock_count
            }

    def test_initialization(self, data_service):
        """Test DataService initializes correctly."""
        assert data_service.cache_enabled is False
        assert data_service._cache is None
        assert data_service._last_fetch is None

    def test_get_data_cached(self, data_service, mock_database, test_data):
        """Test data is cached after first fetch."""
        # Enable caching for this test
        service = DataService(cache_enabled=True)
        mock_database["get"].return_value = test_data
        
        # First call
        result1 = service.get_data()
        mock_database["get"].assert_called_once()
        
        # Second call - should use cache
        result2 = service.get_data()
        # Should not call database again
        mock_database["get"].assert_called_once()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_get_data_not_cached(self, data_service, mock_database, test_data):
        """Test data is not cached when caching disabled."""
        mock_database["get"].return_value = test_data
        
        # First call
        result1 = data_service.get_data()
        mock_database["get"].assert_called_once()
        
        # Second call - should call database again
        result2 = data_service.get_data()
        assert mock_database["get"].call_count == 2
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_get_data_with_filter(self, data_service, mock_database, test_data):
        """Test get_data with filter function."""
        mock_database["get"].return_value = test_data
        
        # Filter for activities longer than 1 hour
        result = data_service.get_data(
            filter_func=lambda df: df[df["duration_hours"] > 1.0]
        )
        
        assert len(result) == len(test_data[test_data["duration_hours"] > 1.0])
        assert all(result["duration_hours"] > 1.0)

    def test_get_data_empty_dataframe(self, data_service, mock_database):
        """Test get_data returns empty DataFrame when no data."""
        mock_database["get"].return_value = pd.DataFrame()
        
        result = data_service.get_data()
        assert result.empty

    def test_get_activity_count(self, data_service, mock_database):
        """Test get_activity_count method."""
        mock_database["count"].return_value = 42
        
        count = data_service.get_activity_count()
        
        assert count == 42
        mock_database["count"].assert_called_once()

    def test_clear_cache(self, data_service, mock_database, test_data):
        """Test clear_cache method."""
        service = DataService(cache_enabled=True)
        mock_database["get"].return_value = test_data
        
        # Fetch data
        service.get_data()
        assert service._last_fetch is not None
        
        # Clear cache
        service.clear_cache()
        assert service._cache is None
        assert service._last_fetch is None

    @pytest.mark.parametrize("cache_enabled", [True, False])
    def test_cache_parameter(self, cache_enabled, mock_database, test_data):
        """Test cache parameter affects caching behavior."""
        service = DataService(cache_enabled=cache_enabled)
        mock_database["get"].return_value = test_data
        
        service.get_data()
        service.get_data()
        
        call_count = mock_database["get"].call_count
        expected_calls = 1 if cache_enabled else 2
        assert call_count == expected_calls