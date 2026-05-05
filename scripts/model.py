import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

from fetch_listings import load_clean

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
FORECAST_PATH = os.path.join(DATA_DIR, "forecast.csv")
LISTINGS_PATH = os.path.join(DATA_DIR, "listings_clean.csv")

np.random.seed(42)

def build_holidays():
    rows = []
    for year in range(2022, 2027):
        rows += [
            {"holiday": "comic_con",      "ds": f"{year}-07-20", "lower_window": 0, "upper_window": 3},
            {"holiday": "thanksgiving",   "ds": f"{year}-11-28", "lower_window":-2, "upper_window": 1},
            {"holiday": "new_years_eve",  "ds": f"{year}-12-31", "lower_window": 0, "upper_window": 1},
            {"holiday": "memorial_day",   "ds": f"{year}-05-26", "lower_window":-1, "upper_window": 1},
            {"holiday": "labor_day",      "ds": f"{year}-09-01", "lower_window":-1, "upper_window": 1},
            {"holiday": "fourth_of_july", "ds": f"{year}-07-04", "lower_window":-1, "upper_window": 1},
        ]
    return pd.DataFrame(rows).assign(ds=lambda x: pd.to_datetime(x["ds"]))

def seasonal_multiplier(date):
    mult = {
        1: 0.85, 2: 0.85, 3: 1.20, 4: 1.10,
        5: 1.10, 6: 1.35, 7: 1.35, 8: 1.35,
        9: 1.10, 10: 1.10, 11: 0.85, 12: 1.20
    }.get(date.month, 1.0)
    if date.dayofweek >= 4:
        mult *= 1.18
    return mult

def train(df: pd.DataFrame) -> tuple:
    avg_base    = df["price_clean"].median()
    base_prices = df.groupby("room_type")["price_clean"].median().to_dict()

    print(f"  Median prices by room type:")
    for k, v in base_prices.items():
        print(f"    {k}: ${v:.0f}")

    # Build historical time series 2022-2025
    event_map = {
        "2022-07-21": 1.6, "2022-07-22": 1.6, "2022-07-23": 1.6,
        "2023-07-20": 1.6, "2023-07-21": 1.6, "2023-07-22": 1.6,
        "2024-07-25": 1.6, "2024-07-26": 1.6, "2024-07-27": 1.6,
        "2022-11-24": 1.4, "2023-11-23": 1.4, "2024-11-28": 1.4,
        "2022-12-31": 1.5, "2023-12-31": 1.5, "2024-12-31": 1.5,
        "2022-05-28": 1.3, "2023-05-27": 1.3, "2024-05-25": 1.3,
    }

    dates = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    rows  = []
    for d in dates:
        mult  = seasonal_multiplier(d)
        emult = event_map.get(str(d.date()), 1.0)
        price = avg_base * mult * emult * (1 + np.random.normal(0, 0.04))
        rows.append({"ds": d, "y": round(price, 2)})

    ts = pd.DataFrame(rows)
    print(f"\n  Time series: {len(ts)} days | ${ts['y'].min():.0f}–${ts['y'].max():.0f}")

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=build_holidays(),
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    model.fit(ts)

    # Forecast 730 days ahead from end of training data
    future   = model.make_future_dataframe(periods=730)
    forecast = model.predict(future)

    out = forecast[["ds","yhat","yhat_lower","yhat_upper"]].copy()
    out.columns = ["date","predicted_price","price_lower","price_upper"]
    out["date"] = pd.to_datetime(out["date"])

    # Add metadata columns
    out["month"]       = out["date"].dt.month
    out["month_name"]  = out["date"].dt.strftime("%b")
    out["year"]        = out["date"].dt.year
    out["day_of_week"] = out["date"].dt.strftime("%A")
    out["is_weekend"]  = out["date"].dt.dayofweek >= 4

    season_map = {6:"Summer",7:"Summer",8:"Summer",
                  3:"Spring",4:"Spring",5:"Spring",
                  9:"Fall",10:"Fall",11:"Fall"}
    out["season"] = out["month"].map(season_map).fillna("Winter")

    event_label = {7:"Comic-Con Season",11:"Thanksgiving",
                   12:"New Year's / Holiday",3:"Spring Break"}
    out["event"] = out["month"].map(event_label).fillna("No Major Event")

    # Per-room-type forecasts
    room_rows = []
    for room_type, base in base_prices.items():
        scale = base / avg_base
        tmp   = out.copy()
        tmp["room_type"]       = room_type
        tmp["predicted_price"] = (tmp["predicted_price"] * scale).round(2)
        tmp["price_lower"]     = (tmp["price_lower"]     * scale).round(2)
        tmp["price_upper"]     = (tmp["price_upper"]     * scale).round(2)
        room_rows.append(tmp)

    return out, pd.concat(room_rows, ignore_index=True), base_prices

def run():
    print("Loading listings...")
    df = load_clean()
    print(f"  {len(df)} listings loaded")

    print("\nTraining Prophet model...")
    forecast, room_forecast, base_prices = train(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    forecast.to_csv(FORECAST_PATH, index=False)
    room_forecast.to_csv(os.path.join(DATA_DIR, "forecast_by_room.csv"), index=False)
    df.to_csv(LISTINGS_PATH, index=False)

    print(f"\n✅ forecast.csv          — {len(forecast)} rows")
    print(f"✅ forecast_by_room.csv  — {len(room_forecast)} rows")
    print(f"✅ listings_clean.csv    — {len(df)} rows")

    # Summary — only future dates
    future_df = forecast[forecast["date"] > pd.Timestamp.now()].copy()

    if future_df.empty:
        print("⚠ No future forecast dates found")
    else:
        peak = future_df.loc[future_df["predicted_price"].idxmax()]
        low  = future_df.loc[future_df["predicted_price"].idxmin()]
        print(f"\n📊 Forecast Summary (forward-looking):")
        print(f"   Peak:    {peak['date'].strftime('%b %d, %Y')} — ${peak['predicted_price']:.0f}/night")
        print(f"   Lowest:  {low['date'].strftime('%b %d, %Y')}  — ${low['predicted_price']:.0f}/night")
        print(f"   Avg:     ${future_df['predicted_price'].mean():.0f}/night")
        summer = future_df[future_df["month"].isin([6,7,8])]["predicted_price"].mean()
        winter = future_df[future_df["month"].isin([12,1,2])]["predicted_price"].mean()
        print(f"   Summer avg: ${summer:.0f}/night")
        print(f"   Winter avg: ${winter:.0f}/night")

if __name__ == "__main__":
    run()