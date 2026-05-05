import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fetch_listings import fetch_latest
from model import run as run_model

def run():
    print("=== Airbnb Pricing Pipeline ===")
    print("\n[1/2] Fetching latest Inside Airbnb data...")
    fetch_latest()

    print("\n[2/2] Training model + generating forecast...")
    run_model()

    print("\n✅ Pipeline complete")

if __name__ == "__main__":
    run()
