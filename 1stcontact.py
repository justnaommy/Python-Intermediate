
import re
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# --- Load CSV (try utf-8 first; if you see â‚¬, switch to encoding="latin1") ---
df = pd.read_csv("./fifa21_raw_data.csv", low_memory=False, encoding="utf-8")

# If you already see 'â‚¬' in your data, uncomment this line instead:
# df = pd.read_csv("./fifa21_raw_data.csv", low_memory=False, encoding="latin1")

# --- Height: convert 5'7" -> inches ---
def convert_height(height_str):
    m = re.match(r"^\s*(\d)'\s*(\d+)\s*\"?\s*$", str(height_str))
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return feet * 12 + inches
    return None

df["Height"] = df["Height"].apply(convert_height)

# --- Weight: "569 lbs" -> 569.0 ---
df["Weight"] = (
    df["Weight"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
)

# --- Strip newlines from all object columns ---
for col in df.select_dtypes(include="object").columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )

# --- Joined: to datetime, then compute tenure in years (vectorized) ---
df["Joined"] = pd.to_datetime(df["Joined"], format="%d-%b-%y", errors="coerce")
today = pd.to_datetime(datetime.today().date())
df["YearsAtClub"] = (today - df["Joined"]).dt.days / 365.0
long_term_df = df.loc[df["YearsAtClub"] > 10, ["Name", "Joined", "YearsAtClub"]].copy()
print(long_term_df.head())

# --- Star ratings like "4★" or "4" -> 4.0 ---
def clean_star_rating(value):
    m = re.search(r"(\d+)", str(value))
    return float(m.group(1)) if m else None

for col in ["SM", "W/F", "IR"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_star_rating)

# --- Money parser: "€120K", "€85M", "€1.2M", "€500" -> numeric euros ---
def parse_money(value):
    if pd.isna(value):
        return None
    s = str(value)
    s = s.replace("€", "").replace("â‚¬", "").strip()  # fix mojibake if any
    m = re.match(r"^\s*([\d\.]+)\s*([KMBkmb]?)\s*$", s)
    if not m:
        # try plain float after removing commas
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None
    num = float(m.group(1))
    suffix = m.group(2).upper()
    mult = 1
    if suffix == "K":
        mult = 1_000
    elif suffix == "M":
        mult = 1_000_000
    elif suffix == "B":
        mult = 1_000_000_000
    return num * mult

for col in ["Value", "Wage", "Release Clause"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_money)

# --- Scatter: Value vs Wage (guard against missing columns) ---
if {"Wage", "Value"}.issubset(df.columns):
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Wage"], df["Value"], alpha=0.5, edgecolors="black")
    plt.xlabel("Wage (€)")
    plt.ylabel("Value (€)")
    plt.title("Value vs Wage: Identifying Undervalued Players")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Columns 'Wage' and/or 'Value' not found; skipping plot.")

# --- Save cleaned data ---
df.to_csv("cleaned_data.csv", index=False, encoding="utf-8")
print("Saved cleaned_data.csv")





