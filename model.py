import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────
np.random.seed(42)

print("Loading data from PostgreSQL...")
engine = create_engine('postgresql://gabrieldiaz@localhost/pricing_engine')
df = pd.read_sql('SELECT * FROM listings', engine)
print(f"Loaded {len(df)} listings")

# ── Base prices by room type ─────────────────────────────────────────────
base_prices = df.groupby('room_type')['nightly_rate'].median().to_dict()
print("\nMedian price by room type:")
for k, v in base_prices.items():
    print(f"  {k}: ${v:.0f}")

# ── Seasonal multiplier ───────────────────────────────────────────────────
def seasonal_multiplier(date):
    m = date.month
    dow = date.dayofweek
    mult = 1.0
    if m in [6, 7, 8]:         mult *= 1.35   # Summer peak
    elif m in [3, 12]:         mult *= 1.20   # Spring break + holidays
    elif m in [4, 5, 9, 10]:   mult *= 1.10   # Shoulder season
    else:                       mult *= 0.85   # Low season (Jan, Feb, Nov)
    if dow >= 4:                mult *= 1.18   # Weekend bump
    return mult

# ── SoCal event multipliers ──────────────────────────────────────────────
events = {
    # Comic-Con San Diego
    '2022-07-21': 1.6, '2022-07-22': 1.6, '2022-07-23': 1.6, '2022-07-24': 1.6,
    '2023-07-20': 1.6, '2023-07-21': 1.6, '2023-07-22': 1.6, '2023-07-23': 1.6,
    '2024-07-25': 1.6, '2024-07-26': 1.6, '2024-07-27': 1.6, '2024-07-28': 1.6,  # fixed: was missing from holidays
    # Thanksgiving week
    '2022-11-24': 1.4, '2022-11-25': 1.4, '2022-11-26': 1.4,
    '2023-11-23': 1.4, '2023-11-24': 1.4, '2023-11-25': 1.4,
    '2024-11-28': 1.4, '2024-11-29': 1.4, '2024-11-30': 1.4,
    # New Year's Eve
    '2022-12-31': 1.5, '2023-12-31': 1.5, '2024-12-31': 1.5,
    # Memorial Day weekend
    '2022-05-28': 1.3, '2022-05-29': 1.3, '2022-05-30': 1.3,
    '2023-05-27': 1.3, '2023-05-28': 1.3, '2023-05-29': 1.3,
    '2024-05-25': 1.3, '2024-05-26': 1.3, '2024-05-27': 1.3,
}

# ── Build aggregate time series ──────────────────────────────────────────
dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
avg_base = df['nightly_rate'].median()
rows = []
for d in dates:
    mult = seasonal_multiplier(d)
    event_mult = events.get(str(d.date()), 1.0)
    noise = np.random.normal(0, 0.04)
    price = avg_base * mult * event_mult * (1 + noise)
    rows.append({'ds': d, 'y': round(price, 2)})

ts = pd.DataFrame(rows)
print(f"\nTime series built: {len(ts)} days")
print(f"Price range: ${ts['y'].min():.0f} – ${ts['y'].max():.0f}")

# ── Prophet holidays (fixed: 2024 Comic-Con now included) ────────────────
holidays = pd.DataFrame({
    'holiday': [
        'comic_con','comic_con','comic_con','comic_con',
        'comic_con','comic_con','comic_con','comic_con',
        'comic_con','comic_con','comic_con','comic_con',   # 2024 added
        'thanksgiving','thanksgiving','thanksgiving',
        'thanksgiving','thanksgiving','thanksgiving',
        'new_years_eve','new_years_eve','new_years_eve',
    ],
    'ds': pd.to_datetime([
        '2022-07-21','2022-07-22','2022-07-23','2022-07-24',
        '2023-07-20','2023-07-21','2023-07-22','2023-07-23',
        '2024-07-25','2024-07-26','2024-07-27','2024-07-28',  # 2024 added
        '2022-11-24','2022-11-25','2022-11-26',
        '2023-11-23','2023-11-24','2023-11-25',
        '2022-12-31','2023-12-31','2024-12-31',
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# ── Train Prophet ────────────────────────────────────────────────────────
print("\nTraining Prophet model...")
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)
m.fit(ts)

# ── Cross-validation & performance metrics ───────────────────────────────
print("\nRunning cross-validation (this may take a moment)...")
df_cv = cross_validation(m, initial='730 days', period='90 days', horizon='180 days')
df_perf = performance_metrics(df_cv)
print("\n📐 MODEL PERFORMANCE (cross-validation):")
print(df_perf[['horizon', 'mape', 'rmse']].iloc[::5].to_string(index=False))  # print every 5th row for brevity

# ── Forecast 12 months ahead ─────────────────────────────────────────────
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# ── Save aggregate forecast to PostgreSQL ────────────────────────────────
forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
forecast_out.columns = ['date', 'predicted_price', 'price_lower', 'price_upper']
forecast_out.to_sql('forecast', engine, if_exists='replace', index=False)
print("\n✅ Aggregate forecast saved to PostgreSQL!")

# ── Per-room-type forecasts ───────────────────────────────────────────────
print("\nGenerating per-room-type forecasts...")
room_forecasts = []
for room_type, base_price in base_prices.items():
    scale = base_price / avg_base
    room_fc = forecast_out.copy()
    room_fc['room_type'] = room_type
    room_fc['predicted_price'] = (room_fc['predicted_price'] * scale).round(2)
    room_fc['price_lower']     = (room_fc['price_lower']     * scale).round(2)
    room_fc['price_upper']     = (room_fc['price_upper']     * scale).round(2)
    room_forecasts.append(room_fc)

room_forecast_df = pd.concat(room_forecasts, ignore_index=True)
room_forecast_df.to_sql('forecast_by_room_type', engine, if_exists='replace', index=False)
print(f"✅ Per-room-type forecast saved ({len(room_forecast_df)} rows across {len(base_prices)} room types)!")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full forecast
axes[0].plot(ts['ds'], ts['y'], alpha=0.5, color='steelblue', linewidth=0.8, label='Historical')
axes[0].plot(forecast['ds'], forecast['yhat'], color='tomato', linewidth=1.5, label='Forecast')
axes[0].fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                     alpha=0.15, color='tomato', label='Confidence interval')
axes[0].axvline(ts['ds'].max(), color='gray', linestyle='--', linewidth=1, label='Forecast start')
axes[0].set_title('San Diego Airbnb — Daily Price Forecast (Prophet)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Predicted Nightly Rate ($)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Average predicted price by month
forecast['month'] = forecast['ds'].dt.month
monthly = forecast.groupby('month')['yhat'].mean()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#d9534f' if v == monthly.max() else '#5bc0de' if v == monthly.min()
          else '#5cb85c' for v in monthly.values]
axes[1].bar(month_names, monthly.values, color=colors, edgecolor='white')
axes[1].set_title('Average Predicted Price by Month — Seasonal Pattern', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Average Nightly Rate ($)')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(monthly.values):
    axes[1].text(i, v + 2, f'${v:.0f}', ha='center', fontsize=9)

plt.tight_layout()
out_path = os.path.abspath('pricing_forecast.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"✅ Chart saved → {out_path}")

# ── Forecast summary ──────────────────────────────────────────────────────
print("\n📊 FORECAST SUMMARY (2025)")
print("=" * 40)
future_forecast = forecast[forecast['ds'] > pd.Timestamp('2024-12-31')]
peak = future_forecast.loc[future_forecast['yhat'].idxmax()]
low  = future_forecast.loc[future_forecast['yhat'].idxmin()]
print(f"Peak date:    {peak['ds'].strftime('%b %d, %Y')} — ${peak['yhat']:.0f}/night")
print(f"Lowest date:  {low['ds'].strftime('%b %d, %Y')} — ${low['yhat']:.0f}/night")
print(f"Avg 2025:     ${future_forecast['yhat'].mean():.0f}/night")
print(f"Summer avg:   ${future_forecast[future_forecast['ds'].dt.month.isin([6,7,8])]['yhat'].mean():.0f}/night")
print(f"Winter avg:   ${future_forecast[future_forecast['ds'].dt.month.isin([12,1,2])]['yhat'].mean():.0f}/night")