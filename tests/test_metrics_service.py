"""
Unit tests for MetricsService
"""
import pytest
import pandas as pd
import numpy as np

from src.services.metrics_service import MetricsService


@pytest.mark.unit
class TestMetricsService:
    """Test MetricsService class."""

    def test_calculate_basic_metrics(self, test_data):
        """Test basic metrics calculation."""
        service = MetricsService()
        metrics = service.calculate_basic_metrics(test_data)
        
        assert "total_hours" in metrics
        assert "deep_work_hours" in metrics
        assert "daily_avg" in metrics
        assert "total_switches" in metrics
        
        assert metrics["total_hours"] > 0
        assert metrics["deep_work_hours"] >= 0
        assert metrics["deep_work_hours"] <= metrics["total_hours"]

    def test_calculate_deep_work_hours(self, test_data):
        """Test deep work hours calculation."""
        service = MetricsService()
        hours = service.calculate_deep_work_hours(test_data)
        
        # Deep work = Work, Coding, Learning
        deep_work_mask = test_data["activity_type"].isin(["Work", "Coding", "Learning"])
        expected = test_data[deep_work_mask]["duration_hours"].sum()
        
        assert hours == pytest.approx(expected, rel=1e-10)

    def test_calculate_daily_average(self, test_data):
        """Test daily average calculation."""
        service = MetricsService()
        avg = service.calculate_daily_average(test_data)
        
        # Should be total hours divided by unique days
        total_hours = test_data["duration_hours"].sum()
        unique_days = test_data["datetime_from"].dt.date.nunique()
        expected = total_hours / unique_days
        
        assert avg == pytest.approx(expected, rel=1e-10)

    def test_calculate_consistency_metrics_empty(self):
        """Test consistency metrics with empty data."""
        service = MetricsService()
        empty_df = pd.DataFrame()
        
        metrics = service.calculate_consistency_metrics(empty_df)
        
        assert metrics is not None
        # Should return empty DataFrame with expected columns
        assert isinstance(metrics, pd.DataFrame)

    def test_calculate_advanced_metrics(self, test_data):
        """Test advanced metrics calculation."""
        service = MetricsService()
        metrics = service.calculate_advanced_metrics(test_data)
        
        assert "deep_work_ratio" in metrics
        assert "flow_index" in metrics
        assert "sleep_regularity" in metrics
        
        # Deep work ratio should be between 0 and 100
        assert 0 <= metrics["deep_work_ratio"] <= 100
        # Flow index should be between 0 and 100
        assert 0 <= metrics["flow_index"] <= 100

    def test_deep_work_ratio_calculation(self, test_data):
        """Test deep work ratio calculation."""
        service = MetricsService()
        ratio = service.calculate_deep_work_ratio(test_data)
        
        total_hours = test_data["duration_hours"].sum()
        deep_work_mask = test_data["activity_type"].isin(["Work", "Coding", "Learning"])
        deep_work_hours = test_data[deep_work_mask]["duration_hours"].sum()
        expected_ratio = (deep_work_hours / total_hours * 100) if total_hours > 0 else 0
        
        assert ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_get_category_summary(self, test_data):
        """Test category summary calculation."""
        service = MetricsService()
        summary = service.get_category_summary(test_data)
        
        assert "activity_type" in summary.columns
        assert "total_hours" in summary.columns
        assert "percentage" in summary.columns
        
        # Percentages should sum to 100
        total_percentage = summary["percentage"].sum()
        assert total_percentage == pytest.approx(100.0, rel=1e-10)

    def test_calculate_all_metrics(self, test_data):
        """Test calculating all metrics at once."""
        service = MetricsService()
        metrics = service.calculate_all_metrics(test_data)
        
        # Should include all metric types
        assert "basic" in metrics
        assert "advanced" in metrics
        assert "consistency" in metrics
        assert "categories" in metrics

    @pytest.mark.parametrize("activity_type,expected_is_deep", [
        ("Work", True),
        ("Coding", True),
        ("Learning", True),
        ("Break", False),
        ("Exercise", False),
        ("Other", False),
    ])
    def test_is_deep_work_classification(self, activity_type, expected_is_deep):
        """Test activity type classification as deep work."""
        service = MetricsService()
        df = pd.DataFrame({
            "activity_type": [activity_type],
            "duration_hours": [1.0],
            "datetime_from": [pd.Timestamp("2026-01-01")],
            "datetime_to": [pd.Timestamp("2026-01-01 01:00:00")],
        })
        
        df = service._add_is_deep_work_flag(df)
        is_deep = df["is_deep_work"].iloc[0]
        
        assert is_deep == (1 if expected_is_deep else 0)

    def test_flow_index_calculation_no_data(self):
        """Test flow index with no deep work data."""
        service = MetricsService()
        no_deep_work_df = pd.DataFrame({
            "activity_type": ["Break", "Other"],
            "duration_hours": [1.0, 1.0],
            "datetime_from": pd.date_range("2026-01-01", periods=2, freq="D"),
            "datetime_to": pd.date_range("2026-01-01", periods=2, freq="D") + pd.Timedelta(hours=1),
        })
        
        flow_index = service.calculate_flow_index(no_deep_work_df)
        
        # Should be 0 when no deep work
        assert flow_index == 0.0