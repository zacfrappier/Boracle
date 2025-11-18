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
#   set seed
np.random.seed(9)

#   Create directory if already doesnt exist
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
# if performs poorly can cahnge this to 7-14
TIME_STEPS = 30

# function definition to create 3D time series sequences from 2D flat dataframe
# also known as "sliding window" technique
# MODIFIED function definition to create 3D time series sequences from separate X and Y data
# This ensures Y is only the single injury label for the next step.
def create_time_series_data(X_data, y_data, time_steps):
    X, y = [], []
    # Ensure both X and y arrays are long enough to form a sequence and next step
    max_len = min(len(X_data), len(y_data))
    for i in range(max_len - time_steps):
        # X is the sequence of features up to time_steps
        X.append(X_data[i:(i + time_steps)]) 
        # y is the target label (injury status) at the NEXT time step
        y.append(y_data[i + time_steps]) 
    return np.array(X), np.array(y)

#                   --- Load Data and Initial Inspection ---
#   Load Dataset
df = pd.read_csv('timeseries (daily).csv')

# Convert the 'Date" column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

#features to be used in model here for reference
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

#probalby wont be used but added here
all_features = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative', 'perceived exertion', 'perceived trainingSuccess', 'perceived recovery',
                'nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 'perceived exertion.1', 'perceived trainingSuccess.1', 'perceived recovery.1', 
                'nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2', 'perceived exertion.2', 'perceived trainingSuccess.2', 'perceived recovery.2', 
                'nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3', 'perceived exertion.3', 'perceived trainingSuccess.3', 'perceived recovery.3', 
                'nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4', 'perceived exertion.4', 'perceived trainingSuccess.4', 'perceived recovery.4', 
                'nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5', 'perceived exertion.5', 'perceived trainingSuccess.5', 'perceived recovery.5', 
                'nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6', 'perceived exertion.6', 'perceived trainingSuccess.6', 'perceived recovery.6',
                'Athlete ID', 'injury', 'Date']
# both objective and subjective data 70 features
raw_features_combined = all_features[:-3]
# only objective data 49 features 
raw_features_objective = ['nr. sessions', 'total km', 'km Z3-4', 'km Z5-T1-T2', 'km sprinting', 'strength training', 'hours alternative',
                'nr. sessions.1', 'total km.1', 'km Z3-4.1', 'km Z5-T1-T2.1', 'km sprinting.1', 'strength training.1', 'hours alternative.1', 
                'nr. sessions.2', 'total km.2', 'km Z3-4.2', 'km Z5-T1-T2.2', 'km sprinting.2', 'strength training.2', 'hours alternative.2',  
                'nr. sessions.3', 'total km.3', 'km Z3-4.3', 'km Z5-T1-T2.3', 'km sprinting.3', 'strength training.3', 'hours alternative.3',  
                'nr. sessions.4', 'total km.4', 'km Z3-4.4', 'km Z5-T1-T2.4', 'km sprinting.4', 'strength training.4', 'hours alternative.4',  
                'nr. sessions.5', 'total km.5', 'km Z3-4.5', 'km Z5-T1-T2.5', 'km sprinting.5', 'strength training.5', 'hours alternative.5',  
                'nr. sessions.6', 'total km.6', 'km Z3-4.6', 'km Z5-T1-T2.6', 'km sprinting.6', 'strength training.6', 'hours alternative.6'] 
#sort data by athlete ID and Date for chronological order for time-series analysis
df = df.sort_values(by=['Athlete ID', 'Date'])

#checks for Nans, missing vlaues, and data types;
#--------   ADD MORE INSPECTION HERE   ------------------------
print('not finished with initial data inspection')
print('number of null values:',df.isnull().sum())

#               --- Preprocessing for Objective Model ---
print('--- Starting preprocessing for Objective Model ---')

#  Feature Engineering for Objective model 
# scaling skipped for singular feature
df['objective_training_load'] = df['total km']

# Rolling Metrics for Objective Training Load for ACWR
df['acute_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['chronic_load_obj'] = df.groupby('Athlete ID')['acute_load_obj'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
df['objective_acwr'] = df['acute_load_obj'] / df['chronic_load_obj']

# Training Load and Monotony into Objective Strain
df['weekly_avg_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['weekly_std_load_obj'] = df.groupby('Athlete ID')['objective_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
df['monotony_obj'] = df['weekly_avg_load_obj'] / df['weekly_std_load_obj']
df['objective_strain'] = df['objective_training_load'] * df['monotony_obj']

#Define features to be used for the objective model
objective_features = ['objective_strain', 
                'objective_acwr'] + raw_features_objective

# Handle NaN and infinite values that arise from calculations
df.replace([np.inf, -np.inf], np.nan, inplace=True) # np.inf = infinity replace with np.nan = 'not a number'
#counter for imputed data
# first sum counts true per col, second counts across columns
imputed_data_objective = df[objective_features].isna().sum().sum()
df.fillna(0, inplace=True) # all nan become '0'

print('Objective Model Imputations')
print(f'Total data points imputed (set to 0): {imputed_data_objective}')

# SEPARATE X (features) and Y (target) data arrays
data_objective_X = df[objective_features].values
data_objective_y = df['injury'].values # 1D target array

# 4. Create Time-Series Sequences & Split Data 

#create sequences 
X_objective, y_objective = create_time_series_data(data_objective_X, data_objective_y, TIME_STEPS)

#Split data into training and validation sets
split_index_obj = int(0.8 * len(X_objective)) #determines split, .8 = 80% train 20% validate
X_objective_train, X_objective_val = X_objective[:split_index_obj], X_objective[split_index_obj:]
y_objective_train, y_objective_val = y_objective[:split_index_obj], y_objective[split_index_obj:]

# transform and normalize 

scaler_objective_X = MinMaxScaler() #first create scalar object, to call on class

# b/c x is 3D, need to flatten
#flatten to 2D
n_train, t, f = X_objective_train.shape
X_train_flat = X_objective_train.reshape(-1, f)

# scaling (fit only on train!)
X_train_flat = scaler_objective_X.fit_transform(X_train_flat)
X_val_flat   = scaler_objective_X.transform(X_objective_val.reshape(-1, f))

# reshape back
X_objective_train = X_train_flat.reshape(n_train, t, f)
X_objective_val   = X_val_flat.reshape(len(X_objective_val), t, f)

#b/c y is label, no need to scale, just use float (0 or 1)
y_objective_train = y_objective_train.astype(float)
y_objective_val = y_objective_val.astype(float)

print(f"Objective model data created with shape: X_train={X_objective_train.shape}, y_train={y_objective_train.shape}")

#Save Preprocessed data and scaler for the objective model 
np.save('preprocessed_data_objective/X_train.npy', X_objective_train)
np.save('preprocessed_data_objective/X_val.npy', X_objective_val)
np.save('preprocessed_data_objective/y_train.npy', y_objective_train)
np.save('preprocessed_data_objective/y_val.npy',y_objective_val)
with open('preprocessed_data_objective/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_objective_X, f)
with open('preprocessed_data_objective/objective_features.pkl', 'wb') as f:
    pickle.dump(objective_features, f)
print("Objective data saved successfully")

#           --- Preprocessing for Combined Model ---
print('\n--- now starting data preprocessing for combined model ---')

# Scaling and Feature Engineering for Combined Model

# Create the combined Training Load Metric
df['combined_training_load'] = df['perceived exertion'] * df['total km']

# Combined ACWR
df['acute_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['chronic_load_combined'] = df.groupby('Athlete ID')['acute_load_combined'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())
df['combined_acwr'] = df['acute_load_combined'] / df['chronic_load_combined']


# Combined Strain
df['weekly_avg_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df['weekly_std_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x : x.rolling(window=7, min_periods=1).std())
df['combined_monotony'] = df['weekly_avg_load_combined'] / df['weekly_std_load_combined']
df['combined_strain'] = df['combined_training_load'] * df['combined_monotony']

#features to be used for subjective model
combined_features = ['combined_monotony', 'combined_strain', 'combined_acwr'] + raw_features_combined

# Handle and record NaN and infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
#counter for imputed values 
# first sum is for true per col, second sum for across cols
imputed_data_combined = df[combined_features].isna().sum().sum()
df.fillna(0, inplace=True)

print("Combined Model Imputations")
print(f'Total data points imputed (set to 0):{imputed_data_combined}')

# SEPARATE X (features) and Y (target) data arrays
data_combined_X = df[combined_features].values
data_combined_y = df['injury'].values # 1D target array

# 4. Creating Time-Series Sequences & 5. Splitting Data
# Scale the combined dataset (note: this is a new scaler, not the one from before)


# Create the sequences using the new function that handles X and y separately
X_combined, y_combined = create_time_series_data(data_combined_X, data_combined_y, TIME_STEPS)

# Split the data into training and validation sets (e.g., 80/20 split)
split_index_combined = int(0.8 * len(X_combined))
X_combined_train, X_combined_val = X_combined[:split_index_combined], X_combined[split_index_combined:]
y_combined_train, y_combined_val = y_combined[:split_index_combined], y_combined[split_index_combined:]
 
# scale data here
scaler_combined_X = MinMaxScaler()

# b/c x is 3D, need to flatten
#flatten to 2D
# flatten to 2D
n_train, t, f = X_combined_train.shape
X_train_flat = X_combined_train.reshape(-1, f)
X_val_flat = X_combined_val.reshape(-1, f)

# scaling (fit only on train!)
X_train_flat = scaler_combined_X.fit_transform(X_train_flat)
X_val_flat   = scaler_combined_X.transform(X_val_flat)

# reshape back
X_combined_train = X_train_flat.reshape(n_train, t, f)
X_combined_val   = X_val_flat.reshape(len(X_combined_val), t, f)

#b/c y is label, no need to scale, just use float (0 or 1)
y_combined_train = y_combined_train.astype(float)
y_combined_val = y_combined_val.astype(float)

print(f"Combined model data created with shapes: X_train={X_combined_train.shape}, y_train={y_combined_train.shape}")

# Save preprocessed data and scaler for the combined model
np.save('preprocessed_data_combined/X_train.npy', X_combined_train)
np.save('preprocessed_data_combined/X_val.npy', X_combined_val)
np.save('preprocessed_data_combined/y_train.npy', y_combined_train)
np.save('preprocessed_data_combined/y_val.npy', y_combined_val)
with open('preprocessed_data_combined/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_combined_X, f)
with open('preprocessed_data_combined/combined_features.pkl', 'wb') as f:
    pickle.dump(combined_features, f)
print("Combined data saved successfully.")