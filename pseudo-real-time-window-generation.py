import pandas as pd

df = pd.read_csv("dataset_for_model_long.csv", parse_dates=["date"])
df = df.set_index("date").sort_index()

start_test = "2020-01-01"
horizons = [1,3,6,12]

forecast_origins = df.loc[start_test:].index

windows = []

for t in forecast_origins:

    train_end = t - pd.DateOffset(months=1)

    train = df.loc[:train_end]

    for h in horizons:

        target_date = t + pd.DateOffset(months=h)

        if target_date in df.index:

            windows.append({
                "forecast_origin": t,
                "train_end": train_end,
                "horizon": h,
                "target_date": target_date
            })

windows = pd.DataFrame(windows)

windows.to_csv("forecast_windows.csv", index=False)

print("Total forecast tasks:", len(windows))