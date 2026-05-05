import pandas as pd
from sqlalchemy import create_engine

print("Loading CSV...")
df = pd.read_csv("listings.csv", low_memory=False)
print(f"Raw data: {len(df)} rows, {len(df.columns)} columns")

# Keep only the columns we need
df = df[[
    'id', 'neighbourhood_cleansed', 'room_type',
    'price', 'minimum_nights', 'number_of_reviews',
    'availability_365'
]].copy()

# ── Clean price column ───────────────────────────────────────────────────
df['price'] = df['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Rename columns before further processing
df = df.rename(columns={
    'id': 'listing_id',
    'neighbourhood_cleansed': 'neighbourhood',
    'price': 'nightly_rate'
})

# ── Handle missing values ────────────────────────────────────────────────
df = df.dropna(subset=['nightly_rate'])
df['number_of_reviews'] = df['number_of_reviews'].fillna(0).astype(int)
df['availability_365'] = df['availability_365'].fillna(df['availability_365'].median())
df['minimum_nights'] = df['minimum_nights'].fillna(1).clip(upper=365).astype(int)

# ── Remove price outliers using quantile cap ─────────────────────────────
lower = 10
upper = df['nightly_rate'].quantile(0.995)
before = len(df)
df = df[(df['nightly_rate'] >= lower) & (df['nightly_rate'] <= upper)]
print(f"Dropped {before - len(df)} outlier rows (price > ${upper:.0f})")

# ── Derive date/time fields dynamically ──────────────────────────────────
# Use today's date as the snapshot date instead of a hardcoded value
snapshot_date = pd.Timestamp.today().normalize()
df['date'] = snapshot_date
df['month'] = snapshot_date.month
df['year'] = snapshot_date.year

# Derive season from actual month
def get_season(m):
    if m in [6, 7, 8]:   return 'Summer'
    elif m in [3, 4, 5]: return 'Spring'
    elif m in [9, 10, 11]: return 'Fall'
    else:                  return 'Winter'

df['season'] = df['month'].apply(get_season)

# Derive is_weekend from the actual snapshot date
df['is_weekend'] = snapshot_date.dayofweek >= 4  # Friday=4, Saturday=5, Sunday=6

# Static fields
df['location'] = 'San Diego, CA'
df['event_flag'] = False

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\nClean data:    {len(df)} rows")
print(f"Snapshot date: {snapshot_date.strftime('%Y-%m-%d')} ({snapshot_date.strftime('%A')})")
print(f"Season:        {get_season(snapshot_date.month)}")
print(f"Price range:   ${df['nightly_rate'].min():.0f} – ${df['nightly_rate'].max():.0f}")
print(f"Average price: ${df['nightly_rate'].mean():.0f}")
print(f"Median price:  ${df['nightly_rate'].median():.0f}")
print(f"\nListings by room type:")
print(df['room_type'].value_counts().to_string())

# ── Load into PostgreSQL ─────────────────────────────────────────────────
engine = create_engine('postgresql://gabrieldiaz@localhost/pricing_engine')
df.to_sql('listings', engine, if_exists='replace', index=False)
print("\n✅ Data loaded into PostgreSQL successfully!")