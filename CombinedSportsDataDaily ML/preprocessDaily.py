import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
from sklearn.preprocessing import MinMaxScalar

# Data Preprocess
#   1. Data Loading and initial inspection
#           missing values, outliers, datatypes
#           conduct imputation, transformation if needed
#   2. Scaling/Normalizing                      
#           scale between 0-1
#           store scaler object for later use of inverse transformation during prediction evaluation
#   3. Feature Engineering (normalize data first before this step)
#           create new features
#           normalize values before making feature to avoid feature overweighing
#   4. Creating Time-Series Sequences - .pkl files
#           GRU requires input data into sequences 
#           transform flat time series data into 3D array
#           'Look-back' window - # timestpes to consider a prediction
#           input-output pairs
#   5. Splitting Data
#           training/testing set 
#           save large numerical data with numpy .npy files
#           save scalar objects with pickle

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
TIME_STEPS = 30

# function definition to create 3D time series sequences from 2D flat dataframe
# also known as "sliding window" technique
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)]) # creates a sequence of 'time_steps' as input (X)
        y.append(data[i + time_steps]) # next data point is the target (y)
    return np.array(X), np.array(y)

#                   --- Load Data and Initial Inspection ---
#   Load Dataset
df = pd.read_csv('timeseries (daily).csv')

# Convert the 'Date" column to a datetime object
df['Date'] = pd.to_datetime(df['Date'])

#sort data by athlete ID and Date for chronological order for time-series analysis
df = df.sort_values(by=['Athlete_ID', 'Date'])

#checks for Nans, missing vlaues, and data types;
#----------------------------------------------------------------ADD MORE INSPECTION HERE ------------------------
print('not finished with initial data inspection')
print(df.isnull().sum())

#               --- Preprocessing for Objective Model ---
print('--- Starting preprocessing for Objective Model ---')

# 2 & 3 Scaling & Feature Engineering for Objective model 
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

# Handle NaN and infinite values that arise from calculations
df.replace([np.inf, -np.inf], np.nan, inplace=True) # np.inf = infinity replace with np.nan = 'not a number'
df.fillna(0, inplace=True) # all nan become '0'

#Define features to be used ofr the objective model
#------------------------------------------------------------ ADD MORE FEATURES HERE-------------------
print('not done with setting features for model to use')
obj_features = ['objective_strain', 'objective_acwr']
data_obj = df[obj_features].values

#4. Create Time-Series Sequences & Split Data 
scaler_obj = MinMaxScalar()
scaled_data_obj = scaler_obj.fit_transform(data_obj)

#create sequences 
X_obj, y_obj = create_sequences(scaled_data_obj, TIME_STEPS)

#Split data into training and validation sets
split_index_obj = int(0.8 * len(X_obj)) #determines split, .8 = 80% train 20% validate
X_obj_train, X_obj_val = X_obj[:split_index_obj], X_obj[split_index_obj:]
y_obj_train, y_obj_val = y_obj[:split_index_obj], y_obj[split_index_obj:]

print(f"Objective model data created with shape: X_train={X_obj_train.shape}, y_train={y_obj_train.shape}")

#Save Preprocessed data and scaler for the objective model 
np.save('preprocessed_data_objective/X_train.npy', X_obj_train)
np.save('preprocessed_data_objective/X_val.npy', X_obj_val)
np.save('preprocessed_data_objective/y_train.npy', y_obj_train)
np.save('preprocessed_data_objective/y_val.npy',y_obj_val)
with open('preprocessed_data_objective/scalar.pkl', 'wb') as f:
    pickle.dump(scaler_obj, f)
print("Objective data saved successfully")

#           --- Preprocessing for Combined Model ---
print('\n--- now starting data preprocessing for combined model ---')

# Scaling and Feature Engineering for Combined Model
# Normalize 'rpe' and 'total km' before combining tem
scaler_combined = MinMaxScalar()
df[['perceived exertion', 'total km']] = scaler_combined.fit_transform(df[['perceived exertion', 'total km']])

# Create the combined Training Load Metric
df['combined_training_load'] = df['perceived exertion'] * df['total km']

# Rolling Metrics from combined training load
df['acute_load_combined'] = df.groupby('Athlete ID')['combined_training_load'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
df['chronic_load_combined'] = df.groupby('Athlete.ID')['acute_load_combined'].transform(lambda x: x.rolling(window=28, min_periods=1).mean())

