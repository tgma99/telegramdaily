python - <<'PY'
import pandas as pd
from datetime import datetime, timezone, timedelta

csv_path = "/Users/tomadshead/telegramdaily/data/filtered/filtered_2025-12-26.csv"
df = pd.read_csv(csv_path)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# Parse datetime robustly
dt_raw = df["datetime"] if "datetime" in df.columns else None
if dt_raw is None:
    print("❌ No 'datetime' column found.")
    raise SystemExit(1)

dt = pd.to_datetime(dt_raw, errors="coerce", utc=True)
print("Parsed datetime NaT:", int(dt.isna().sum()), "of", len(dt))

now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=24)

in_24h = dt >= cutoff
print("Now (UTC):", now.isoformat())
print("Cutoff:", cutoff.isoformat())
print("In last 24h:", int(in_24h.sum()))

print("Min dt:", dt.min())
print("Max dt:", dt.max())

# Country distribution within last 24h (if present)
if "country" in df.columns:
    d2 = df.loc[in_24h].copy()
    print("\nCountries in last 24h:")
    print(d2["country"].fillna("unknown").value_counts().head(20))
else:
    print("\n⚠️ No 'country' column found in CSV.")
PY