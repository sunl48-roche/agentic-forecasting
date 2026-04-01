"""Tests for SeriesStore and DataService."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.data.store import SeriesStore


def _make_meta(series_id: str = "test_series") -> SeriesMetadata:
    return SeriesMetadata(
        series_id=series_id,
        description="Test series",
        source="test",
        units="Index",
        frequency="MS",
    )


def _make_df(timestamps: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})


def _make_adapter(df: pd.DataFrame) -> MagicMock:
    adapter = MagicMock()
    adapter.fetch.return_value = df
    return adapter


class TestSeriesStore:
    def setup_method(self) -> None:
        self.store = SeriesStore()

    def test_put_and_get_roundtrip(self) -> None:
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        self.store.put("s1", df, _make_meta("s1"))
        pd.testing.assert_frame_equal(self.store.get("s1"), df)

    def test_get_returns_copy(self) -> None:
        """Mutations on the returned DataFrame must not affect the store."""
        df = _make_df(["2022-01-01"], [100.0])
        self.store.put("s1", df, _make_meta("s1"))
        copy = self.store.get("s1")
        copy.loc[0, "value"] = 999.0
        assert self.store.get("s1")["value"].iloc[0] == 100.0

    def test_get_unknown_series_raises(self) -> None:
        with pytest.raises(KeyError, match="not_a_series"):
            self.store.get("not_a_series")

    def test_put_validates_required_columns(self) -> None:
        bad_df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-01-01"])})
        with pytest.raises(ValueError, match="value"):
            self.store.put("s1", bad_df, _make_meta("s1"))


class TestDataService:
    def setup_method(self) -> None:
        self.svc = DataService()

    def test_register_and_get_series_with_cutoff(self) -> None:
        """End-to-end: register series, retrieve with cutoff applied."""
        df = _make_df(["2022-01-01", "2022-02-01", "2022-03-01"], [100.0, 101.0, 102.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        result = self.svc.get_series("s1", as_of=datetime(2022, 2, 1))
        assert list(result["value"]) == [100.0, 101.0]

    def test_get_series_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            self.svc.get_series("not_registered", as_of=datetime(2022, 1, 1))

    def test_get_metadata(self) -> None:
        df = _make_df(["2022-01-01"], [1.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        assert self.svc.get_metadata("s1").source == "test"

    def test_summary_structure(self) -> None:
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        summary = self.svc.summary()
        assert {"series_id", "n_obs", "start", "end"}.issubset(summary.columns)
        assert summary.loc[0, "n_obs"] == 2
