from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ==============================
# CONFIG
# ==============================
ZIP_PATH = Path("/mnt/data/data.zip")
OUTPUT_CSV = Path("/mnt/data/master_table_russia_2001_01_long_sample.csv")
OUTPUT_XLSX = Path("/mnt/data/master_table_russia_2001_01_long_sample.xlsx")
START_DATE = "2001-01-01"
END_DATE = None  # None -> last available date in the data

# Default factor set for the agreed long-sample design
SELECTED_COLUMNS = [
    "cpi_mm",                    # target, monthly CPI inflation in p.p. (index-100)
    "m2_eom",                    # M2, end of month level
    "m2_log_mom",                # 100 * dlog(M2)
    "m2_log_yoy",                # 100 * 12m dlog(M2)
    "miacr_1d_avg",              # monthly average of daily MIACR 1-day
    "usd_rub_avg",               # monthly average RUB/USD
    "usd_rub_log_mom",           # 100 * dlog(monthly average RUB/USD)
    "brent_avg",                 # monthly average Brent
    "brent_log_mom",             # 100 * dlog(monthly average Brent)
    "business_climate_cbr",      # Bank of Russia business climate indicator
    "unemployment_rate",         # unemployment level, %
    "nominal_wage",              # monthly nominal wage level
    "wage_log_yoy",              # 100 * 12m dlog(wage)
    "ppi_industry_mm",           # industrial PPI m/m, p.p. (index-100)
    "ppi_construction_mm",       # construction PPI m/m, p.p. (index-100)
    "ffpi_food",                 # FAO food price index level
    "ffpi_food_log_mom",         # 100 * dlog(FAO food index)
]

RU_MONTHS_FULL = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}

RU_MONTHS_SHORT = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "май": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
}


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace("\xa0", " ")
        if x in {"-", "—", "…", "..", ".", ""}:
            return np.nan
        x = x.replace(",", ".")
        x = re.sub(r"\s+", "", x)
        # drop footnotes like '2022 1)' or values with text tails
        x = re.sub(r"(?<=\d)\)$", "", x)
        m = re.search(r"[-+]?\d*\.?\d+", x)
        if m:
            return float(m.group())
        return np.nan
    return pd.to_numeric(x, errors="coerce")


def parse_year(x):
    if pd.isna(x):
        return None
    s = str(x)
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group()) if m else None


def parse_month_name_ru(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower().replace("ё", "е")
    s = s.split()[0]
    return RU_MONTHS_FULL.get(s)


def parse_mon_yy_ru(s: str) -> pd.Timestamp:
    s = str(s).strip().lower().replace("ё", "е")
    m = re.match(r"([а-я]{3})\.?([0-9]{2})", s)
    if not m:
        raise ValueError(f"Cannot parse month-year token: {s}")
    mon = RU_MONTHS_SHORT[m.group(1)]
    yy = int(m.group(2))
    year = 2000 + yy if yy <= 69 else 1900 + yy
    return pd.Timestamp(year=year, month=mon, day=1)


def read_excel_from_zip(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    data = io.BytesIO(zf.read(name))
    return pd.read_excel(data, engine="openpyxl", **kwargs)


def make_month_index(start_date: str, end_date: str | None = None) -> pd.DatetimeIndex:
    end = end_date or pd.Timestamp.today().strftime("%Y-%m-01")
    return pd.date_range(start=start_date, end=end, freq="MS")


# ------------------------------
# Parsers for each source type
# ------------------------------
def parse_simple_monthly(zf: zipfile.ZipFile, filename: str, value_col: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.set_index(pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp())[value_col]
    s = s[~s.index.isna()]
    s.name = out_col
    return s.sort_index()


def parse_daily_to_monthly_avg(zf: zipfile.ZipFile, filename: str, date_col: str, value_col: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = df[value_col].map(clean_numeric)
    df = df.dropna(subset=[date_col])
    s = df.set_index(date_col)[value_col].resample("MS").mean()
    s.name = out_col
    return s.sort_index()


def parse_wide_year_month_table(
    zf: zipfile.ZipFile,
    filename: str,
    sheet_name: str,
    year_row: int,
    month_start_row: int,
    month_col: int,
    first_year_col: int,
    out_col: str,
) -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name=sheet_name, header=None)
    years = []
    for c in range(first_year_col, df.shape[1]):
        y = parse_year(df.iloc[year_row, c])
        years.append((c, y))
    years = [(c, y) for c, y in years if y is not None]

    records = []
    for r in range(month_start_row, df.shape[0]):
        month = parse_month_name_ru(df.iloc[r, month_col])
        if month is None:
            continue
        for c, year in years:
            val = clean_numeric(df.iloc[r, c])
            records.append((pd.Timestamp(year=year, month=month, day=1), val))

    out = pd.DataFrame(records, columns=["date", out_col]).dropna(subset=["date"])
    out = out.groupby("date", as_index=True)[out_col].first().sort_index()
    out.name = out_col
    return out


def parse_monitoring_business_climate(zf: zipfile.ZipFile, filename: str, out_col: str = "business_climate_cbr") -> pd.Series:
    df = read_excel_from_zip(zf, filename, header=None)
    header_row = None
    for i in range(min(20, len(df))):
        row = df.iloc[i].astype(str).tolist()
        if any("Индикатор бизнес-климата Банка России" == str(x).strip() for x in row):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Header row not found in Monitoring file")
    data = df.iloc[header_row + 1 :].copy()
    data.columns = df.iloc[header_row].tolist()
    data = data.rename(columns={data.columns[0]: "date_token"})
    data["date"] = data["date_token"].apply(parse_mon_yy_ru)
    data[out_col] = data["Индикатор бизнес-климата Банка России"].map(clean_numeric)
    s = data.set_index("date")[out_col].sort_index()
    s.name = out_col
    return s


def parse_unemployment(zf: zipfile.ZipFile, filename: str, out_col: str = "unemployment_rate") -> pd.Series:
    df = read_excel_from_zip(zf, filename, header=None)
    data = df.iloc[2:, [0, 1, 2]].copy()
    data.columns = ["year", "month_name", out_col]
    data["year"] = data["year"].ffill().map(clean_numeric)
    data["month"] = data["month_name"].map(parse_month_name_ru)
    data[out_col] = data[out_col].map(clean_numeric)
    data = data.dropna(subset=["year", "month"])
    data["date"] = pd.to_datetime(
        dict(year=data["year"].astype(int), month=data["month"].astype(int), day=1)
    )
    s = data.set_index("date")[out_col].sort_index()
    s.name = out_col
    return s


def parse_wage(zf: zipfile.ZipFile, filename: str, out_col: str = "nominal_wage") -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name="Лист1", header=None)
    years = df.iloc[7:, 0].map(parse_year)
    month_cols = {
        6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6,
        12: 7, 13: 8, 14: 9, 15: 10, 16: 11, 17: 12,
    }
    records = []
    for r in range(7, df.shape[0]):
        year = parse_year(df.iloc[r, 0])
        if year is None:
            continue
        for c, month in month_cols.items():
            if c >= df.shape[1]:
                continue
            val = clean_numeric(df.iloc[r, c])
            if pd.isna(val):
                continue
            records.append((pd.Timestamp(year=year, month=month, day=1), val))
    out = pd.DataFrame(records, columns=["date", out_col]).drop_duplicates(subset=["date"])
    s = out.set_index("date")[out_col].sort_index()
    s.name = out_col
    return s


def parse_ffpi(zf: zipfile.ZipFile, filename: str, out_col: str = "ffpi_food") -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name="Indices_Monthly_Nominal")
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].sort_index()
    s = s.groupby(s.index.to_period("M").to_timestamp()).first()
    s.name = out_col
    return s


# ------------------------------
# Derived transforms
# ------------------------------
def dlog(series: pd.Series, periods: int = 1) -> pd.Series:
    return 100 * (np.log(series) - np.log(series.shift(periods)))


def build_master_table(zip_path: Path = ZIP_PATH) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        # 1) Target: CPI m/m -> from sheet 01, values are indices to previous month, convert to p.p.
        cpi_index = parse_wide_year_month_table(
            zf=zf,
            filename="инфляция и индекс продовольствия.xlsx",
            sheet_name="01",
            year_row=3,
            month_start_row=5,
            month_col=0,
            first_year_col=1,
            out_col="cpi_index_mm",
        )

        # 2) M2: monthly level, end of month
        m2 = parse_simple_monthly(zf, "M2.xlsx", value_col="Всего", out_col="m2_eom")

        # 3) MIACR: daily -> monthly average, 1-day tenor
        miacr = parse_daily_to_monthly_avg(zf, "MIACR.xlsx", date_col="Дата", value_col="1 день", out_col="miacr_1d_avg")

        # 4) USD/RUB: daily -> monthly average
        usd = parse_daily_to_monthly_avg(zf, "курс рубля к доллару.xlsx", date_col="data", value_col="curs", out_col="usd_rub_avg")

        # 5) Brent: daily -> monthly average
        brent = parse_daily_to_monthly_avg(zf, "нефть-brent.xlsx", date_col="Дата", value_col="Значение", out_col="brent_avg")

        # 6) Business climate indicator
        bci = parse_monitoring_business_climate(zf, "Мониторинг предприятий.xlsx", out_col="business_climate_cbr")

        # 7) Unemployment
        ur = parse_unemployment(zf, "безработица.xlsx", out_col="unemployment_rate")

        # 8) Nominal wage
        wage = parse_wage(zf, "зарплата.xlsx", out_col="nominal_wage")

        # 9) Industrial PPI m/m from sheet 1.1, convert index->p.p.
        ppi_ind = parse_wide_year_month_table(
            zf=zf,
            filename="Индексы цен производителей по ОКВЭД. Промышленные товары.xlsx",
            sheet_name="1.1",
            year_row=2,
            month_start_row=4,
            month_col=0,
            first_year_col=1,
            out_col="ppi_industry_index_mm",
        )

        # 10) Construction PPI m/m from sheet 1, convert index->p.p.
        ppi_constr = parse_wide_year_month_table(
            zf=zf,
            filename="Индексы цен производителей. Строительная продукция.xlsx",
            sheet_name="1",
            year_row=5,
            month_start_row=7,
            month_col=0,
            first_year_col=1,
            out_col="ppi_construction_index_mm",
        )

        # 11) FAO Food Price Index
        ffpi = parse_ffpi(zf, "FFPI(мировые цены на продовольствие).xlsx", out_col="ffpi_food")

    month_index = make_month_index(START_DATE, END_DATE)
    master = pd.DataFrame(index=month_index)
    master.index.name = "date"

    # Join aligned raw monthly series
    for s in [cpi_index, m2, miacr, usd, brent, bci, ur, wage, ppi_ind, ppi_constr, ffpi]:
        master = master.join(s, how="left")

    # Convert index-style inflation series to monthly inflation in p.p.
    master["cpi_mm"] = master["cpi_index_mm"] - 100
    master["ppi_industry_mm"] = master["ppi_industry_index_mm"] - 100
    master["ppi_construction_mm"] = master["ppi_construction_index_mm"] - 100

    # Derived transforms for model-ready columns
    master["m2_log_mom"] = dlog(master["m2_eom"], 1)
    master["m2_log_yoy"] = dlog(master["m2_eom"], 12)
    master["usd_rub_log_mom"] = dlog(master["usd_rub_avg"], 1)
    master["brent_log_mom"] = dlog(master["brent_avg"], 1)
    master["wage_log_yoy"] = dlog(master["nominal_wage"], 12)
    master["ffpi_food_log_mom"] = dlog(master["ffpi_food"], 1)

    # Keep requested default columns first, then append raw helper columns
    helper_cols = [c for c in master.columns if c not in SELECTED_COLUMNS]
    ordered = [c for c in SELECTED_COLUMNS if c in master.columns] + helper_cols
    master = master[ordered]

    # Trim to end date if provided
    if END_DATE is not None:
        master = master.loc[:END_DATE]

    return master


def main() -> None:
    master = build_master_table(ZIP_PATH)
    master.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    master.to_excel(OUTPUT_XLSX)

    print("Master table created")
    print(f"Rows: {len(master)}")
    print(f"Columns: {len(master.columns)}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"XLSX: {OUTPUT_XLSX}")
    print("\nCoverage preview:")
    coverage = master.notna().sum().sort_values(ascending=False)
    print(coverage.to_string())
    print("\nHead:")
    print(master.head(12).to_string())
    print("\nTail:")
    print(master.tail(12).to_string())


if __name__ == "__main__":
    main()
