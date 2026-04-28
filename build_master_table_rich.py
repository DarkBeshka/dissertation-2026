from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ZIP_PATH = Path('/mnt/data/data.zip')
OUTPUT_CSV = Path('/mnt/data/master_table_russia_2014_01_rich_model.csv')
OUTPUT_XLSX = Path('/mnt/data/master_table_russia_2014_01_rich_model.xlsx')
START_DATE = '2014-01-01'
END_DATE = None

SELECTED_COLUMNS = [
    # target
    'cpi_mm',
    # money aggregates
    'm0_eom', 'm0_log_mom',
    'm1_eom', 'm1_log_mom',
    'm2_eom', 'm2_log_mom', 'm2_log_yoy',
    'm2x_eom', 'm2x_log_mom',
    # rates and financial conditions
    'miacr_1d_avg', 'ruonia_avg', 'key_rate_avg',
    'corp_credit_rate_gt1y', 'hh_credit_rate_gt1y',
    'corp_deposit_rate_gt1y', 'hh_deposit_rate_gt1y',
    # FX / commodities / reserves
    'usd_rub_avg', 'usd_rub_log_mom',
    'brent_avg', 'brent_log_mom',
    'reserves_eom', 'reserves_log_mom',
    # real sector and labor
    'business_climate_cbr', 'unemployment_rate',
    'nominal_wage', 'wage_log_yoy',
    # producer / transport / global food prices
    'ppi_industry_mm', 'ppi_construction_mm', 'ppi_agriculture_mm', 'cargo_freight_mm',
    'ffpi_food', 'ffpi_food_log_mom',
    # expectations
    'infl_exp_observed_median', 'infl_exp_expected_median',
    # budget (monthly flow from cumulative YTD)
    'budget_revenue_flow', 'budget_oilgas_revenue_flow', 'budget_nonoil_revenue_flow',
    'budget_expenditure_flow', 'budget_balance_flow',
]

RU_MONTHS_FULL = {
    'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6,
    'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12,
}
RU_MONTHS_SHORT = {
    'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6,
    'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
}


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace('\xa0', ' ')
        if x in {'-', '—', '…', '..', '.', '', 'nan'}:
            return np.nan
        x = x.replace(',', '.')
        x = re.sub(r'\s+', '', x)
        x = re.sub(r'(?<=\d)\)$', '', x)
        m = re.search(r'[-+]?\d*\.?\d+', x)
        return float(m.group()) if m else np.nan
    return pd.to_numeric(x, errors='coerce')


def parse_year(x):
    if pd.isna(x):
        return None
    m = re.search(r'(19|20)\d{2}', str(x))
    return int(m.group()) if m else None


def parse_month_name_ru(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower().replace('ё', 'е').split()[0]
    return RU_MONTHS_FULL.get(s)


def read_excel_from_zip(zf: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(zf.read(name)), engine='openpyxl', **kwargs)


def dlog(series: pd.Series, periods: int = 1) -> pd.Series:
    return 100 * (np.log(series) - np.log(series.shift(periods)))


def make_month_index(start_date: str, end_date: str | None = None) -> pd.DatetimeIndex:
    end = end_date or pd.Timestamp.today().strftime('%Y-%m-01')
    return pd.date_range(start=start_date, end=end, freq='MS')


def parse_simple_monthly(zf: zipfile.ZipFile, filename: str, value_col: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(df[date_col].dt.to_period('M').dt.to_timestamp())[value_col]
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.name = out_col
    return s


def parse_daily_to_monthly_avg(zf: zipfile.ZipFile, filename: str, date_col: str, value_col: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].resample('MS').mean()
    s.name = out_col
    return s.sort_index()


def parse_daily_to_monthly_last(zf: zipfile.ZipFile, filename: str, date_col: str, value_col: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].resample('MS').last()
    s.name = out_col
    return s.sort_index()


def parse_wide_year_month_table(zf: zipfile.ZipFile, filename: str, sheet_name: str, year_row: int, month_start_row: int, month_col: int, first_year_col: int, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name=sheet_name, header=None)
    years = []
    for c in range(first_year_col, df.shape[1]):
        y = parse_year(df.iloc[year_row, c])
        if y is not None:
            years.append((c, y))
    records = []
    for r in range(month_start_row, df.shape[0]):
        month = parse_month_name_ru(df.iloc[r, month_col])
        if month is None:
            continue
        for c, year in years:
            val = clean_numeric(df.iloc[r, c])
            records.append((pd.Timestamp(year=year, month=month, day=1), val))
    out = pd.DataFrame(records, columns=['date', out_col]).dropna(subset=['date'])
    s = out.groupby('date', as_index=True)[out_col].first().sort_index()
    s.name = out_col
    return s


def parse_monitoring_business_climate(zf: zipfile.ZipFile, filename: str, out_col: str='business_climate_cbr') -> pd.Series:
    df = read_excel_from_zip(zf, filename, header=None)
    header_row = None
    for i in range(min(25, len(df))):
        row = [str(x).strip() for x in df.iloc[i].tolist()]
        if 'Индикатор бизнес-климата Банка России' in row:
            header_row = i
            break
    if header_row is None:
        raise ValueError('Header row not found in monitoring file')
    data = df.iloc[header_row+1:].copy()
    data.columns = df.iloc[header_row].tolist()
    date_col = data.columns[0]
    data['date'] = data[date_col].astype(str).str.strip().str.lower().str.replace('ё','е')
    data['date'] = data['date'].str.extract(r'([а-я]{3}\.?\d{2})', expand=False)
    def parse_mon_yy_ru(s):
        if pd.isna(s):
            return pd.NaT
        m = re.match(r'([а-я]{3})\.?([0-9]{2})', str(s))
        if not m:
            return pd.NaT
        mon = RU_MONTHS_SHORT[m.group(1)]
        yy = int(m.group(2))
        year = 2000 + yy if yy <= 69 else 1900 + yy
        return pd.Timestamp(year=year, month=mon, day=1)
    data['date'] = data['date'].apply(parse_mon_yy_ru)
    data[out_col] = data['Индикатор бизнес-климата Банка России'].map(clean_numeric)
    s = data.dropna(subset=['date']).set_index('date')[out_col].sort_index()
    s.name = out_col
    return s


def parse_unemployment(zf: zipfile.ZipFile, filename: str, out_col: str='unemployment_rate') -> pd.Series:
    df = read_excel_from_zip(zf, filename, header=None)
    data = df.iloc[2:, [0,1,2]].copy()
    data.columns = ['year', 'month_name', out_col]
    data['year'] = data['year'].ffill().map(clean_numeric)
    data['month'] = data['month_name'].map(parse_month_name_ru)
    data[out_col] = data[out_col].map(clean_numeric)
    data = data.dropna(subset=['year','month'])
    data['date'] = pd.to_datetime(dict(year=data['year'].astype(int), month=data['month'].astype(int), day=1))
    s = data.set_index('date')[out_col].sort_index()
    s.name = out_col
    return s


def parse_wage(zf: zipfile.ZipFile, filename: str, out_col: str='nominal_wage') -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name='Лист1', header=None)
    month_cols = {6:1,7:2,8:3,9:4,10:5,11:6,12:7,13:8,14:9,15:10,16:11,17:12}
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
    s = pd.DataFrame(records, columns=['date', out_col]).drop_duplicates('date').set_index('date')[out_col].sort_index()
    s.name = out_col
    return s


def parse_ffpi(zf: zipfile.ZipFile, filename: str, out_col: str='ffpi_food') -> pd.Series:
    df = read_excel_from_zip(zf, filename, sheet_name='Indices_Monthly_Nominal', header=1)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    value_col = 'Food Price Index' if 'Food Price Index' in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = df[value_col].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(date_col)[value_col].resample('MS').first()
    s.name = out_col
    return s.sort_index()


def parse_rate_file(zf: zipfile.ZipFile, filename: str, col_name: str, out_col: str) -> pd.Series:
    df = read_excel_from_zip(zf, filename)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[col_name] = df[col_name].map(clean_numeric)
    s = df.dropna(subset=[date_col]).set_index(df[date_col].dt.to_period('M').dt.to_timestamp())[col_name]
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.name = out_col
    return s


def parse_budget_monthly_flows(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
    df = read_excel_from_zip(zf, filename, sheet_name='месяц', header=None)
    date_cols = []
    for c in range(2, df.shape[1]):
        val = df.iloc[2, c]
        if isinstance(val, (pd.Timestamp, __import__('datetime').datetime)):
            date_cols.append((c, pd.Timestamp(val).to_period('M').to_timestamp()))
    # rows of interest
    row_map = {
        'budget_revenue_cum': 4,
        'budget_oilgas_revenue_cum': 5,
        'budget_nonoil_revenue_cum': 6,
        'budget_expenditure_cum': 18,
        'budget_balance_cum': 34,
    }
    out = pd.DataFrame(index=pd.DatetimeIndex([d for _, d in date_cols]))
    out.index.name = 'date'
    for col_name, row_num in row_map.items():
        values = [clean_numeric(df.iloc[row_num, c]) for c, _ in date_cols]
        out[col_name] = values
    # convert cumulative-within-year to monthly flow
    for src, dst in [
        ('budget_revenue_cum', 'budget_revenue_flow'),
        ('budget_oilgas_revenue_cum', 'budget_oilgas_revenue_flow'),
        ('budget_nonoil_revenue_cum', 'budget_nonoil_revenue_flow'),
        ('budget_expenditure_cum', 'budget_expenditure_flow'),
        ('budget_balance_cum', 'budget_balance_flow'),
    ]:
        out[dst] = np.nan
        for yr, grp in out.groupby(out.index.year):
            vals = grp[src].astype(float)
            out.loc[grp.index, dst] = vals.diff()
            if len(vals):
                out.loc[grp.index[0], dst] = vals.iloc[0]
    return out.sort_index()


def latest_inflation_expectations_file(zf: zipfile.ZipFile) -> str:
    candidates = []
    pat = re.compile(r'infl_exp_(\d{2})-(\d{2})\.xlsx$', re.I)
    for name in zf.namelist():
        m = pat.search(name)
        if m:
            yy, mm = int(m.group(1)), int(m.group(2))
            year = 2000 + yy
            candidates.append((year, mm, name))
    if not candidates:
        raise FileNotFoundError('No inflation expectations files found')
    return sorted(candidates)[-1][2]


def parse_inflation_expectations(zf: zipfile.ZipFile) -> pd.DataFrame:
    name = latest_inflation_expectations_file(zf)
    df = read_excel_from_zip(zf, name, sheet_name='Данные за все годы', header=None)
    dates = pd.to_datetime(df.iloc[1, 1:].tolist(), errors='coerce')
    obs = pd.Series(df.iloc[69, 1:].map(clean_numeric).tolist(), index=dates, name='infl_exp_observed_median')
    exp = pd.Series(df.iloc[70, 1:].map(clean_numeric).tolist(), index=dates, name='infl_exp_expected_median')
    out = pd.concat([obs, exp], axis=1)
    out.index = pd.to_datetime(out.index).to_period('M').to_timestamp()
    out = out[~out.index.isna()]
    out = out.groupby(out.index).last().sort_index()
    return out


def build_master_table(zip_path: Path = ZIP_PATH) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        cpi_index = parse_wide_year_month_table(zf, 'инфляция и индекс продовольствия.xlsx', '01', 3, 5, 0, 1, 'cpi_index_mm')

        m0 = parse_simple_monthly(zf, 'M0.xlsx', 'Денежный агрегат М0', 'm0_eom')
        m1 = parse_simple_monthly(zf, 'M1.xlsx', 'Всего', 'm1_eom')
        m2 = parse_simple_monthly(zf, 'M2.xlsx', 'Всего', 'm2_eom')
        m2x = parse_simple_monthly(zf, 'Широкая денежная масса.xlsx', 'Всего', 'm2x_eom')

        miacr = parse_daily_to_monthly_avg(zf, 'MIACR.xlsx', 'Дата', '1 день', 'miacr_1d_avg')
        ruonia = parse_daily_to_monthly_avg(zf, 'RUONIA.xlsx', 'DT', 'ruo', 'ruonia_avg')
        key_rate = parse_daily_to_monthly_avg(zf, 'Ключевая ставка.xlsx', 'Дата', 'Ставка', 'key_rate_avg')

        corp_dep = parse_rate_file(zf, 'Ставки по депозитам нефинансовых организаций.xlsx', 'Свыше 1 года', 'corp_deposit_rate_gt1y')
        hh_dep = parse_rate_file(zf, 'Ставки по депозитам физических лиц.xlsx', 'Свыше 1 года', 'hh_deposit_rate_gt1y')
        corp_cred = parse_rate_file(zf, 'Ставки по кредитам нефинансовым организациям.xlsx', 'Свыше 1 года', 'corp_credit_rate_gt1y')
        hh_cred = parse_rate_file(zf, 'Ставки по кредитам физическим лицам.xlsx', 'Свыше 1 года', 'hh_credit_rate_gt1y')

        usd = parse_daily_to_monthly_avg(zf, 'курс рубля к доллару.xlsx', 'data', 'curs', 'usd_rub_avg')
        brent = parse_daily_to_monthly_avg(zf, 'нефть-brent.xlsx', 'Дата', 'Значение', 'brent_avg')
        reserves = parse_daily_to_monthly_last(zf, 'Международные резервы.xlsx', 'Дата', 'Международные резервы', 'reserves_eom')

        bci = parse_monitoring_business_climate(zf, 'Мониторинг предприятий.xlsx')
        ur = parse_unemployment(zf, 'безработица.xlsx')
        wage = parse_wage(zf, 'зарплата.xlsx')

        ppi_ind = parse_wide_year_month_table(zf, 'Индексы цен производителей по ОКВЭД. Промышленные товары.xlsx', '1.1', 2, 4, 0, 1, 'ppi_industry_index_mm')
        ppi_constr = parse_wide_year_month_table(zf, 'Индексы цен производителей. Строительная продукция.xlsx', '1', 5, 7, 0, 1, 'ppi_construction_index_mm')
        ppi_agri = parse_wide_year_month_table(zf, 'Индексы цен проивзодителей. Сельскохозяйственная продукция.xlsx', '1.1', 3, 5, 0, 1, 'ppi_agriculture_index_mm')
        cargo = parse_wide_year_month_table(zf, 'Индексы цен на грузовые перевозки.xlsx', '1', 3, 5, 0, 1, 'cargo_freight_index_mm')

        ffpi = parse_ffpi(zf, 'FFPI(мировые цены на продовольствие).xlsx')
        infl_exp = parse_inflation_expectations(zf)
        budget = parse_budget_monthly_flows(zf, 'Исполнение бюджета.xlsx')

    month_index = make_month_index(START_DATE, END_DATE)
    master = pd.DataFrame(index=month_index)
    master.index.name = 'date'
    
    for obj in [cpi_index, m0, m1, m2, m2x, miacr, ruonia, key_rate, corp_dep, hh_dep, corp_cred, hh_cred,
                usd, brent, reserves, bci, ur, wage, ppi_ind, ppi_constr, ppi_agri, cargo, ffpi]:
        master = master.join(obj, how='left')
    master = master.join(infl_exp, how='left')
    master = master.join(budget, how='left')

    # Derived transforms
    master['cpi_mm'] = master['cpi_index_mm'] - 100
    master['ppi_industry_mm'] = master['ppi_industry_index_mm'] - 100
    master['ppi_construction_mm'] = master['ppi_construction_index_mm'] - 100
    master['ppi_agriculture_mm'] = master['ppi_agriculture_index_mm'] - 100
    master['cargo_freight_mm'] = master['cargo_freight_index_mm'] - 100

    master['m0_log_mom'] = dlog(master['m0_eom'])
    master['m1_log_mom'] = dlog(master['m1_eom'])
    master['m2_log_mom'] = dlog(master['m2_eom'])
    master['m2_log_yoy'] = dlog(master['m2_eom'], 12)
    master['m2x_log_mom'] = dlog(master['m2x_eom'])
    master['usd_rub_log_mom'] = dlog(master['usd_rub_avg'])
    master['brent_log_mom'] = dlog(master['brent_avg'])
    master['reserves_log_mom'] = dlog(master['reserves_eom'])
    master['wage_log_yoy'] = dlog(master['nominal_wage'], 12)
    master['ffpi_food_log_mom'] = dlog(master['ffpi_food'])

    ordered = [c for c in SELECTED_COLUMNS if c in master.columns] + [c for c in master.columns if c not in SELECTED_COLUMNS]
    master = master[ordered]
    return master


def main() -> None:
    master = build_master_table(ZIP_PATH)
    master.to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    master.to_excel(OUTPUT_XLSX)
    print('Master table created')
    print(f'Rows: {len(master)}')
    print(f'Columns: {len(master.columns)}')
    print(f'CSV: {OUTPUT_CSV}')
    print(f'XLSX: {OUTPUT_XLSX}')
    print('\nCoverage preview:')
    print(master.notna().sum().sort_values(ascending=False).to_string())
    print('\nHead:')
    print(master.head(6).to_string())
    print('\nTail:')
    print(master.tail(6).to_string())

if __name__ == '__main__':
    main()
