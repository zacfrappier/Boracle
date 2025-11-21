import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScaler

# Data Preprocess
#   1. Data Loading and initial inspection
#           missing values, outliers, datatypes
#           conduct imputation, transformation if needed
#   2. Feature Engineering (normalize data first before this step)
#           create new features
#           normalize values before making feature to avoid feature overweighing
#   3. Creating Time-Series Sequences - .pkl files
#           GRU requires input data into sequences 
#           transform flat time series data into 3D array
#           'Look-back' window - # timestpes to consider a prediction
#           input-output pairs
#   4. Splitting Data
#           training/testing set 
#           save large numerical data with numpy .npy files
#           save scalar objects with pickle
#   5. Scaling/Normalizing                      
#           scale between 0-1
#           store scaler object for later use of inverse transformation during prediction evaluation

#                   --- Global Configuration ---
np.random.seed(9) 
TIME_STEPS = 30
TRAIN_FRAC = 0.8 

#   Create directory if doesnt exist
os.makedirs('preprocessed_data_objective', exist_ok=True)
os.makedirs('preprocessed_data_combined', exist_ok=True)

# Define the look-back window for time series sequences
#   number of prior steps the GRU sees to predict the next time step
#   1) Defines the input shape - dictates the input data matrix
#       30 = 30 days for any single prediction
#       larger number increases input layer, increases number parameters model to learn = more time& memory
#   2) Controls Historical Context - determines duration of historical context for predictions
#       Small Window - model will learn short term dependencies
#       Large Window - more conext, older data likely to become noise

#   choice of 30 b/c acwr metric, '7' may be better?
# if performs poorly can change this to 7-14

# -------------- helper functions --------------
def create_sequences_for_group(X_arr, y_arr, time_steps):
    """Create sliding-window sequences for a single contiguous series (athlete).
       Returns X_seqs (n_seq, time_steps, n_feat), y_seq (n_seq,)
    """
    Xs, ys = [], []
    max_len = min(len(X_arr), len(y_arr))
    for i in range(max_len - time_steps):
        Xs.append(X_arr[i:(i + time_steps)])
        ys.append(y_arr[i + time_steps])
    if len(Xs) == 0:
        return np.empty((0, time_steps, X_arr.shape[1])), np.empty((0,), dtype=int)
    return np.array(Xs), np.array(ys)

def per_athlete_train_val_split(df, feature_cols, target_col, time_steps, train_frac=0.8):
    """For each athlete, create sequences and split them in time order into train/val.
       Returns concatenated X_train, X_val, y_train, y_val (3D arrays for Xs).
    """
    X_train_list, X_val_list = [], []
    y_train_list, y_val_list = [], []

    # Group by athlete in chronological order
    for athlete, g in df.groupby('Athlete ID'):
        g = g.sort_values('Date')
        X_arr = g[feature_cols].values
        y_arr = g[target_col].values
        X_seqs, y_seqs = create_sequences_for_group(X_arr, y_arr, time_steps)
        if X_seqs.shape[0] == 0:
            continue
        split_idx = int(train_frac * X_seqs.shape[0])
        # if split_idx==0, all sequences go to val; handle gracefully
        if split_idx > 0:
            X_train_list.append(X_seqs[:split_idx])
            y_train_list.append(y_seqs[:split_idx])
        if split_idx < X_seqs.shape[0]:
            X_val_list.append(X_seqs[split_idx:])
            y_val_list.append(y_seqs[split_idx:])

    if len(X_train_list) == 0:
        X_train = np.empty((0, time_steps, len(feature_cols)))
        y_train = np.empty((0,), dtype=int)
    else:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0).astype(int)

    if len(X_val_list) == 0:
        X_val = np.empty((0, time_steps, len(feature_cols)))
        y_val = np.empty((0,), dtype=int)
    else:
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0).astype(int)

    return X_train, X_val, y_train, y_val

def scale_3d_train_val(X_train, X_val):
    """Flatten train & val (time dim merged with samples), fit scaler on train, transform both,
       and reshape back to 3D."""
    if X_train.size == 0:
        return X_train, X_val, None
    n_train, t, f = X_train.shape
    n_val = 0 if X_val.size == 0 else X_val.shape[0]

    train_flat = X_train.reshape(-1, f)
    val_flat = X_val.reshape(-1, f) if n_val > 0 else np.empty((0, f))

    scaler = MinMaxScaler()
    train_flat_scaled = scaler.fit_transform(train_flat)
    val_flat_scaled = scaler.transform(val_flat) if n_val > 0 else val_flat

    X_train_scaled = train_flat_scaled.reshape(n_train, t, f)
    X_val_scaled = val_flat_scaled.reshape(n_val, t, f) if n_val > 0 else X_val
    return X_train_scaled, X_val_scaled, scaler


#                   --- Load Data and Initial Inspection ---
df = pd.read_csv('timeseries (daily).csv')      #load dataset
df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date" column to a datetime object
#sort data by athlete ID and Date for chronological order for time-series analysis
df = df.sort_values(['Athlete ID', 'Date']).reset_index(drop=True) 

#additional sets of features incase needed
journal_features_objective = [
    'km Z5-T1-T2.6',
    'km Z5-T1-T2.4',
    'km sprinting.5',
    'nr. sessions.5',
    'strength training.6',
    'km Z3-4.1',
    'nr. sessions.2',
    'km Z5-T1-T2.5',
    'km Z3-4.3',
    'total km.1',
    'hours alternative.4',
    'hours alternative.6'
]
journal_features = [
    'km Z5-T1-T2.6',
    'perceived trainingSuccess.6',
    'perceived recovery',
    'perceived exertion.4',
    'perceived exertion.3',
    'km Z5-T1-T2.4',
    'km sprinting.5',
    'nr. sessions.5',
    'strength training.6',
    'km Z3-4.1',
    'perceived recovery.1',
    'nr. sessions.2',
    'km Z5-T1-T2.5',
    'km Z3-4.3',
    'perceived trainingSuccess.4',
    'total km.1',
    'hours alternative.4',
    'perceived recovery.6',
    'perceived recovery.3',
    'hours alternative.6'
]
all_features = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery',
                'nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 'perceived exertion.1', 'perceived trainingSuccess.1', 'perceived recovery.1', 
                'nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2', 'perceived exertion.2', 'perceived trainingSuccess.2', 'perceived recovery.2', 
                'nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3', 'perceived exertion.3', 'perceived trainingSuccess.3', 'perceived recovery.3', 
                'nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4', 'perceived exertion.4', 'perceived trainingSuccess.4', 'perceived recovery.4', 
                'nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5', 'perceived exertion.5', 'perceived trainingSuccess.5', 'perceived recovery.5', 
                'nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6', 'perceived exertion.6', 'perceived trainingSuccess.6', 'perceived recovery.6',
                'Athlete ID', 'injury', 'Date']

raw_features_combined = all_features[:-3] # both objective and subjective data 70 features minus name, date, sn
# only objective data 49 features 
raw_features_objective = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative',
                'nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 
                'nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2',  
                'nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3',  
                'nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4',  
                'nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5',  
                'nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6'] 

#checks for Nans, missing vlaues, and data types;
# *************** ADD MORE INSPECTION HERE ****************************
print('not finished with initial data inspection')
print('number of null values:',df.isnull().sum()) #null counts per column 

# -------- OBJECTIVE FEATURE ENGINEERING ----------
print('--- Objective feature engineering ---')
# objective training load
df['objective_training_load'] = df['total km']

# rolling metrics (per athlete)
df['acute_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['chronic_load_obj'] = df.groupby('Athlete ID')['acute_load_obj'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
df['objective_acwr'] = df['acute_load_obj'] / (df['chronic_load_obj'] + 1e-9)

df['weekly_avg_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['weekly_std_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
# avoid division by zero
df['monotony_obj'] = df['weekly_avg_load_obj'] / (df['weekly_std_load_obj'].replace(0, np.nan))
df['objective_strain'] = df['objective_training_load'] * df['monotony_obj']

objective_features = ['objective_strain','objective_acwr'] + raw_features_objective

# -------- COMBINED FEATURE ENGINEERING ----------
print('--- Combined feature engineering ---')
# combined training load (use raw per-athlete values; do NOT global-scale here)
df['combined_training_load'] = df['perceived exertion'] * df['total km']

df['acute_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['chronic_load_combined'] = df.groupby('Athlete ID')['acute_load_combined'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
df['combined_acwr'] = df['acute_load_combined'] / (df['chronic_load_combined'] + 1e-9)

df['weekly_avg_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['weekly_std_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
df['combined_monotony'] = df['weekly_avg_load_combined'] / (df['weekly_std_load_combined'].replace(0, np.nan))
df['combined_strain'] = df['combined_training_load'] * df['combined_monotony']

combined_features = ['combined_monotony','combined_strain','combined_acwr'] + raw_features_combined

# -------- HANDLE INF/NA sensibly ----------
# Replace inf with NaN then perform per-athlete forward/back fill, then global fallback
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# forward fill then backward fill per athlete for numeric columns to avoid zeros skewing early windows
numeric_cols = list(set(objective_features + combined_features))
numeric_cols = [c for c in numeric_cols if c in df.columns]  # keep only present columns

df[numeric_cols] = df.groupby('Athlete ID')[numeric_cols].transform(lambda g: g.ffill().bfill())

# any remaining NaNs (very start of athlete history) fill with column median
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# report imputation counts (for logging)
imputed_counts = df[numeric_cols].isnull().sum().sum()  # should be zero now
print(f"Remaining NaNs after imputation (should be 0): {imputed_counts}")

# -------- PREPARE OBJECTIVE DATASETS ----------
print('--- Creating objective sequences and splits ---')
data_objective_X = df[objective_features].values
data_objective_y = df['injury'].values.astype(int)

X_obj_train, X_obj_val, y_obj_train, y_obj_val = per_athlete_train_val_split(
    df, objective_features, 'injury', TIME_STEPS, train_frac=TRAIN_FRAC
)

print("Objective initial shapes (before scaling):")
print("X_train:", X_obj_train.shape, "X_val:", X_obj_val.shape, "y_train:", y_obj_train.shape, "y_val:", y_obj_val.shape)

# scale 3D arrays (fit on train_flat only)
X_obj_train_s, X_obj_val_s, scaler_obj = scale_3d_train_val(X_obj_train, X_obj_val)

# Save objective outputs
np.save('preprocessed_data_objective/X_train.npy', X_obj_train_s)
np.save('preprocessed_data_objective/X_val.npy', X_obj_val_s)
np.save('preprocessed_data_objective/y_train.npy', y_obj_train.astype(int))
np.save('preprocessed_data_objective/y_val.npy', y_obj_val.astype(int))

with open('preprocessed_data_objective/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_obj, f)
with open('preprocessed_data_objective/objective_features.pkl', 'wb') as f:
    pickle.dump(objective_features, f)

print("Objective data saved. Train sequences:", X_obj_train_s.shape[0], "Val sequences:", X_obj_val_s.shape[0])

# -------- PREPARE COMBINED DATASETS ----------
print('--- Creating combined sequences and splits ---')
X_comb_train, X_comb_val, y_comb_train, y_comb_val = per_athlete_train_val_split(
    df, combined_features, 'injury', TIME_STEPS, train_frac=TRAIN_FRAC
)

print("Combined initial shapes (before scaling):")
print("X_train:", X_comb_train.shape, "X_val:", X_comb_val.shape, "y_train:", y_comb_train.shape, "y_val:", y_comb_val.shape)

X_comb_train_s, X_comb_val_s, scaler_comb = scale_3d_train_val(X_comb_train, X_comb_val)

# Save combined outputs
np.save('preprocessed_data_combined/X_train.npy', X_comb_train_s)
np.save('preprocessed_data_combined/X_val.npy', X_comb_val_s)
np.save('preprocessed_data_combined/y_train.npy', y_comb_train.astype(int))
np.save('preprocessed_data_combined/y_val.npy', y_comb_val.astype(int))

with open('preprocessed_data_combined/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_comb, f)
with open('preprocessed_data_combined/combined_features.pkl', 'wb') as f:
    pickle.dump(combined_features, f)

print("Combined data saved. Train sequences:", X_comb_train_s.shape[0], "Val sequences:", X_comb_val_s.shape[0])

print("Preprocessing finished.")