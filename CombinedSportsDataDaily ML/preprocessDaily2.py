# preprocessDaily.py
"""
Preprocessing tailored to the supplied CSV where each row
already contains lagged columns for a fixed number of past days.

Outputs (in ./preprocessed_data/):
 - X_train.npy, X_val.npy : arrays (N, TIME_STEPS, N_FEATURES)
 - y_train.npy, y_val.npy : arrays (N,)
 - scaler_train.pkl, scaler_combined.pkl : scalers (trained on train set only)
Diagnostics printed to stdout.
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

CSV_PATH = "timeseries (daily).csv"
OUT_DIR = "preprocessed_data2"
os.makedirs(OUT_DIR, exist_ok=True)

# Config
LABEL_COL = "injury"
ATHLETE_COL = "Athlete ID"
DATE_COL = "Date"
# Predict horizon: 1 = predict next day injury. Change to 7 to predict next 7 days (requires adjusting logic).
LABEL_SHIFT = 1

# --- load CSV
df = pd.read_csv(CSV_PATH)
print("Loaded CSV:", CSV_PATH, "shape:", df.shape)
# Ensure rows sorted by Athlete and Date (Date in your file is anonymised - treat as increasing)
df = df.sort_values([ATHLETE_COL, DATE_COL]).reset_index(drop=True)

# --- identify base features that have lagged columns, ignore Athlete/label/Date
ignore_cols = {ATHLETE_COL, LABEL_COL, DATE_COL}
all_cols = list(df.columns)
# build mapping base_name -> list of columns (e.g., 'total km' -> ['total km.6', ... 'total km'])
import re
bases = {}
for c in all_cols:
    if c in ignore_cols:
        continue
    m = re.match(r'^(.*?)(?:\.(\d+))?$', c)
    base = m.group(1)
    suffix = m.group(2)
    bases.setdefault(base, []).append((suffix, c))

# Only keep bases that have consistent suffix counts (i.e., lag features)
valid_bases = {b:v for b,v in bases.items() if len(v) >= 1}
print("Feature bases found (examples):", list(valid_bases.keys())[:10])

# Discover TIME_STEPS from a representative base (we assume consistent)
rep_base = next(iter(valid_bases))
suffixes = sorted([int(s) if s is not None else 0 for s,_ in valid_bases[rep_base]])
max_suffix = max(suffixes)
TIME_STEPS = max_suffix + 1
print("Detected TIME_STEPS (lag depth):", TIME_STEPS, "from base feature:", rep_base)

# Build ordered time-step column lists for each base feature
# We'll order time steps from oldest -> most recent (increasing suffix maybe maps to older or newer depending on file; commonly suffix=0 is current)
# We choose order: highest suffix -> ... -> 0    (suffix=max is oldest) -> so sequence[0] = oldest
def get_ordered_cols_for_base(base):
    cols = valid_bases[base]  # list of (suffix_str_or_None, colname)
    # map suffix None -> 0
    cols2 = []
    for s,c in cols:
        ss = int(s) if s is not None else 0
        cols2.append((ss,c))
    # sort by suffix ascending (0..max) and then reverse so oldest -> newest if desired
    cols2_sorted = sorted(cols2, key=lambda x: x[0])
    # We'll assume suffix 0 = current, suffix larger = older. So to get oldest->current:
    cols_ordered = [c for (_,c) in cols2_sorted[::-1]]  # reverse
    return cols_ordered

# Build final feature list and their columns per timestep
feature_bases = sorted(valid_bases.keys())
print("Number of base features discovered:", len(feature_bases))
# remove any weird base that equals Athlete or label by mistake
feature_bases = [b for b in feature_bases if b not in ignore_cols]

# For each base we get TIME_STEPS columns; check consistency
for b in feature_bases[:10]:
    col_list = get_ordered_cols_for_base(b)
    if len(col_list) != TIME_STEPS:
        print("Warning: base", b, "has", len(col_list), "columns (expected", TIME_STEPS, ")")

# Now build X: for each row, construct (TIME_STEPS, N_features) by stacking each base's lag cols
n_rows = len(df)
n_features = len(feature_bases)
print("n_rows:", n_rows, "n_features (base count):", n_features)

# Build a 3D array: rows x TIME_STEPS x features
X = np.zeros((n_rows, TIME_STEPS, n_features), dtype=float)
for fi, base in enumerate(feature_bases):
    cols = get_ordered_cols_for_base(base)  # oldest->current
    # If columns count != TIME_STEPS, pad or trim (unlikely)
    if len(cols) < TIME_STEPS:
        # pad with zeros on left (oldest)
        pad = TIME_STEPS - len(cols)
        padded = [None]*pad + cols
        cols = padded
    for t in range(TIME_STEPS):
        col = cols[t]
        if col is None:
            X[:, t, fi] = 0.0
        else:
            X[:, t, fi] = df[col].values.astype(float)

# Build label: shift injury forward by LABEL_SHIFT within each athlete
df['_label_next'] = df.groupby(ATHLETE_COL)[LABEL_COL].shift(-LABEL_SHIFT)
# Drop any rows where next label is NaN (end-of-athlete)
valid_mask = ~df['_label_next'].isna()
if valid_mask.sum() < len(df):
    print("Dropping", len(df)-valid_mask.sum(), "rows due to no future label for the chosen label shift.")
X = X[valid_mask.values]
y = df.loc[valid_mask, '_label_next'].astype(int).values
athlete_ids = df.loc[valid_mask, ATHLETE_COL].values

print("After shifting labels: X.shape:", X.shape, "y.shape:", y.shape, "positive rate:", y.mean())

# Option: drop rows where the sequence contains only zeros or is all-imputed - here we keep all rows but you can filter.

# Train/val split by athlete (to avoid leakage)
unique_athletes = np.unique(athlete_ids)
rng = np.random.RandomState(42)
rng.shuffle(unique_athletes)
n_train_ath = int(0.8 * len(unique_athletes))
train_ath = set(unique_athletes[:n_train_ath])
train_mask = np.array([aid in train_ath for aid in athlete_ids])

X_train = X[train_mask]
y_train = y[train_mask]
X_val = X[~train_mask]
y_val = y[~train_mask]

print("Train samples:", X_train.shape[0], "Val samples:", X_val.shape[0])
print("Train positive rate:", y_train.mean(), "Val positive rate:", y_val.mean())

# Scale features: fit scaler on training data only (flatten time and samples)
ns, ts, nf = X_train.shape
scaler = MinMaxScaler()
X_train_2d = X_train.reshape(-1, nf)
X_val_2d = X_val.reshape(-1, nf)
scaler.fit(X_train_2d)
X_train_scaled = scaler.transform(X_train_2d).reshape(ns, ts, nf)
X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape[0], ts, nf)

# Save arrays and scaler
np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val_scaled)
np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)
with open(os.path.join(OUT_DIR, 'scaler_train.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata
meta = {
    "TIME_STEPS": TIME_STEPS,
    "FEATURE_BASES": feature_bases,
    "n_features": n_features,
    "label_shift": LABEL_SHIFT,
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Preprocessing complete. Files saved to:", OUT_DIR)
print("Metadata:", meta)
