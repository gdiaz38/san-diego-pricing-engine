import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://gabrieldiaz@localhost/pricing_engine')

# ── Load tables ───────────────────────────────────────────────────────────
listings = pd.read_sql('SELECT * FROM listings', engine)
forecast = pd.read_sql('SELECT * FROM forecast', engine)

# ── Export 1: Forecast with seasonal labels ───────────────────────────────
forecast['date'] = pd.to_datetime(forecast['date'])
forecast['month'] = forecast['date'].dt.month
forecast['month_name'] = forecast['date'].dt.strftime('%b')
forecast['year'] = forecast['date'].dt.year
forecast['day_of_week'] = forecast['date'].dt.strftime('%A')
forecast['is_weekend'] = forecast['date'].dt.dayofweek >= 4

def get_season(m):
    if m in [6, 7, 8]:     return 'Summer'
    elif m in [3, 4, 5]:   return 'Spring'
    elif m in [9, 10, 11]: return 'Fall'
    else:                   return 'Winter'

def get_event(d):
    events = {
        7:  'Comic-Con Season',
        11: 'Thanksgiving',
        12: "New Year's Eve / Holiday",
        3:  'Spring Break',
    }
    return events.get(d.month, 'No Major Event')

forecast['season'] = forecast['month'].apply(get_season)
forecast['event'] = forecast['date'].apply(get_event)
forecast['predicted_price'] = forecast['predicted_price'].round(2)
forecast['price_lower'] = forecast['price_lower'].round(2)
forecast['price_upper'] = forecast['price_upper'].round(2)
forecast['price_range'] = (forecast['price_upper'] - forecast['price_lower']).round(2)

forecast.to_csv('tableau_forecast.csv', index=False)
print(f"✅ Forecast export: {len(forecast)} rows → tableau_forecast.csv")

# ── Export 2: Listings summary by neighbourhood ────────────────────────────
neighbourhood_summary = listings.groupby(['neighbourhood', 'room_type']).agg(
    avg_price=('nightly_rate', 'mean'),
    median_price=('nightly_rate', 'median'),
    listing_count=('listing_id', 'count'),
    avg_availability=('availability_365', 'mean'),
    avg_reviews=('number_of_reviews', 'mean')
).reset_index()

neighbourhood_summary['avg_price'] = neighbourhood_summary['avg_price'].round(2)
neighbourhood_summary['median_price'] = neighbourhood_summary['median_price'].round(2)
neighbourhood_summary['avg_availability'] = neighbourhood_summary['avg_availability'].round(1)
neighbourhood_summary['avg_reviews'] = neighbourhood_summary['avg_reviews'].round(1)
neighbourhood_summary['location'] = 'San Diego, CA'

# Price tier for Tableau map coloring
neighbourhood_summary['price_tier'] = pd.cut(
    neighbourhood_summary['median_price'],
    bins=[0, 100, 175, 250, 400, 9999],
    labels=['Budget', 'Moderate', 'Mid-Range', 'Premium', 'Luxury']
).astype(str)

neighbourhood_summary.to_csv('tableau_neighbourhoods.csv', index=False)
print(f"✅ Neighbourhood export: {len(neighbourhood_summary)} rows → tableau_neighbourhoods.csv")

# ── Export 3: Monthly peak summary (fixed lambda bug) ─────────────────────
# Previously used a lambda referencing the outer `forecast` DataFrame, which
# caused incorrect weekend/weekday splits. Now computed with explicit pre-split groupbys.

weekend_avg = (
    forecast[forecast['is_weekend']]
    .groupby(['year', 'month'])['predicted_price']
    .mean()
    .rename('weekend_avg')
    .round(2)
)
weekday_avg = (
    forecast[~forecast['is_weekend']]
    .groupby(['year', 'month'])['predicted_price']
    .mean()
    .rename('weekday_avg')
    .round(2)
)

monthly = forecast.groupby(['year', 'month', 'month_name', 'season']).agg(
    avg_predicted=('predicted_price', 'mean'),
    max_predicted=('predicted_price', 'max'),
    min_predicted=('predicted_price', 'min'),
).reset_index().round(2)

monthly = (
    monthly
    .join(weekend_avg, on=['year', 'month'])
    .join(weekday_avg, on=['year', 'month'])
)
monthly['weekend_premium'] = (monthly['weekend_avg'] - monthly['weekday_avg']).round(2)

monthly.to_csv('tableau_monthly.csv', index=False)
print(f"✅ Monthly export: {len(monthly)} rows → tableau_monthly.csv")

# ── Export 4: Combined denormalized flat file (new) ───────────────────────
# Joins neighbourhood-level listing stats with the monthly market average price.
# Avoids brittle Tableau data source relationships and adds a price_vs_market ratio.

monthly_market = (
    forecast.groupby('month')['predicted_price']
    .mean()
    .reset_index()
    .rename(columns={'predicted_price': 'market_avg_price'})
    .round({'market_avg_price': 2})
)

# Get the modal month per neighbourhood (most common listing month)
if 'month' in listings.columns:
    neighbourhood_month = (
        listings.groupby('neighbourhood')['month']
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
else:
    # Fallback: use snapshot month
    neighbourhood_month = neighbourhood_summary[['neighbourhood']].drop_duplicates().copy()
    neighbourhood_month['month'] = pd.Timestamp.today().month

combined = (
    neighbourhood_summary
    .merge(neighbourhood_month, on='neighbourhood', how='left')
    .merge(monthly_market, on='month', how='left')
)
combined['price_vs_market'] = (combined['avg_price'] / combined['market_avg_price']).round(3)

combined.to_csv('tableau_combined.csv', index=False)
print(f"✅ Combined export: {len(combined)} rows → tableau_combined.csv")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n📁 FILES READY FOR TABLEAU:")
print("  tableau_forecast.csv       — full daily forecast (line chart, event filter)")
print("  tableau_neighbourhoods.csv — price by neighbourhood (bar chart + map)")
print("  tableau_monthly.csv        — monthly summary with weekend/weekday split (heatmap)")
print("  tableau_combined.csv       — denormalized join for flat data source (avoids Tableau joins)")