import numpy as np
import pandas as pd
import pytest

import build_master_table as bmt


@pytest.mark.parametrize(
    "value,expected",
    [
        ("123", 123.0),
        ("123,45", 123.45),
        ("1 234,56", 1234.56),
        ("-", np.nan),
        ("", np.nan),
        ("2022 1)", 2022.0),
        ("abc 77.5 xyz", 77.5),
    ],
)
def test_clean_numeric(value, expected):
    result = bmt.clean_numeric(value)

    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024", 2024),
        ("Year 1999", 1999),
        ("abc", None),
    ],
)
def test_parse_year(value, expected):
    assert bmt.parse_year(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("январь", 1),
        ("февраль", 2),
        ("декабрь", 12),
        ("не месяц", None),
    ],
)
def test_parse_month_name_ru(value, expected):
    assert bmt.parse_month_name_ru(value) == expected


def test_parse_mon_yy_ru():
    ts = bmt.parse_mon_yy_ru("янв24")

    assert ts.year == 2024
    assert ts.month == 1


def test_parse_mon_yy_ru_old_century():
    ts = bmt.parse_mon_yy_ru("дек99")

    assert ts.year == 1999
    assert ts.month == 12


def test_parse_mon_yy_ru_invalid():
    with pytest.raises(ValueError):
        bmt.parse_mon_yy_ru("bad")


def test_make_month_index():
    idx = bmt.make_month_index(
        "2024-01-01",
        "2024-03-01",
    )

    assert len(idx) == 3
    assert idx[0] == pd.Timestamp("2024-01-01")
    assert idx[-1] == pd.Timestamp("2024-03-01")


def test_dlog():
    s = pd.Series([100, 110])

    result = bmt.dlog(s)

    expected = 100 * (np.log(110) - np.log(100))

    assert np.isnan(result.iloc[0])
    assert pytest.approx(result.iloc[1]) == expected


def test_parse_simple_monthly(monkeypatch):
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-02-01"],
            "value": [10, 20],
        }
    )

    monkeypatch.setattr(
        bmt,
        "read_excel_from_zip",
        lambda *a, **k: df,
    )

    result = bmt.parse_simple_monthly(
        None,
        "x.xlsx",
        "value",
        "series",
    )

    assert result.name == "series"
    assert result.iloc[0] == 10
    assert result.iloc[1] == 20


def test_parse_daily_to_monthly_avg(monkeypatch):
    df = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-15",
                "2024-02-01",
            ],
            "value": [10, 20, 30],
        }
    )

    monkeypatch.setattr(
        bmt,
        "read_excel_from_zip",
        lambda *a, **k: df,
    )

    result = bmt.parse_daily_to_monthly_avg(
        None,
        "x.xlsx",
        "date",
        "value",
        "avg",
    )

    assert result.loc["2024-01-01"] == 15
    assert result.loc["2024-02-01"] == 30
