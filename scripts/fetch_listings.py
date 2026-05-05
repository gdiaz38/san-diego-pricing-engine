import requests, os, gzip, shutil
from bs4 import BeautifulSoup
import pandas as pd

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
BASE_URL  = "https://data.insideairbnb.com/united-states/ca/san-diego"

def get_latest_url():
    r     = requests.get("http://insideairbnb.com/get-the-data/")
    soup  = BeautifulSoup(r.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True)
             if "san-diego" in a["href"] and "data/listings.csv.gz" in a["href"]]
    if not links:
        raise RuntimeError("Could not find San Diego listings URL")
    return links[0]

def fetch_latest():
    os.makedirs(DATA_DIR, exist_ok=True)
    url     = get_latest_url()
    print(f"📥 Fetching: {url}")

    gz_path  = os.path.join(DATA_DIR, "listings_raw.csv.gz")
    out_path = os.path.join(DATA_DIR, "listings.csv")

    r = requests.get(url, stream=True)
    with open(gz_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)

    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    os.remove(gz_path)
    print(f"✅ Saved to {out_path}")
    return out_path

def load_clean():
    path = os.path.join(DATA_DIR, "listings.csv")

    # Fall back to repo root listings.csv if data/ version missing
    if not os.path.exists(path):
        root = os.path.join(os.path.dirname(__file__), "..", "listings.csv")
        if os.path.exists(root):
            print("⚠ Using existing root listings.csv")
            path = root
        else:
            raise FileNotFoundError("No listings.csv found — run fetch_latest() first")

    df = pd.read_csv(path, low_memory=False)

    # Clean price → numeric
    df["price_clean"] = (
        df["price"]
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

    # Drop extreme outliers (< $20 or > $2000)
    df = df[df["price_clean"].between(20, 2000)].copy()

    # Clean review score
    df["review_scores_rating"] = pd.to_numeric(
        df["review_scores_rating"], errors="coerce")

    keep = [
        "id", "neighbourhood_cleansed", "room_type",
        "price_clean", "accommodates", "bedrooms", "beds",
        "availability_365", "number_of_reviews",
        "review_scores_rating", "latitude", "longitude",
        "instant_bookable", "host_is_superhost",
        "estimated_revenue_l365d"
    ]
    return df[[c for c in keep if c in df.columns]]

if __name__ == "__main__":
    fetch_latest()
    df = load_clean()
    print(f"Loaded {len(df)} listings")
    print(df.groupby("room_type")["price_clean"].median())
