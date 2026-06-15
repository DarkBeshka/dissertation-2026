import numpy as np
import pandas as pd
import pytest

import prepare_dataset_long as mdl


def test_safe_log_diff():
    s = pd.Series([100, 110])

    result = mdl.safe_log_diff(s)

    assert np.isnan(result.iloc[0])

    expected = np.log(110) - np.log(100)

    assert pytest.approx(result.iloc[1]) == expected


def test_safe_log_diff_nonpositive():
    s = pd.Series([100, 0, -1])

    result = mdl.safe_log_diff(s)

    assert result.isna().sum() >= 2


def test_main(monkeypatch):
    source = pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=15,
                freq="MS",
            ),
            "cpi_mm": range(15),
            "m2_eom": range(100, 115),
            "usd_rub_avg": range(90, 105),
            "brent_avg": range(70, 85),
            "nominal_wage": range(1000, 1015),
            "ffpi_food": range(120, 135),
        }
    )

    saved = {}

    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *a, **k: source.copy(),
    )

    monkeypatch.setattr(
        pd.DataFrame,
        "to_excel",
        lambda *a, **k: None,
    )

    def fake_csv(self, *args, **kwargs):
        saved["df"] = self.copy()

    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        fake_csv,
    )

    mdl.main()

    out = saved["df"]

    assert "cpi_lag1" in out.columns
    assert "cpi_lag12" in out.columns
    assert "dlog_m2_eom" in out.columns
