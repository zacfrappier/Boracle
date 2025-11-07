import numpy as np

# script to inspect shapes, labels, and prediction variability across runs
# X_train.shape should be (N_smaples, seg_length, N_features)
# y_train.mean() = smol, we have class imbalance
# np.unique(y_train) - need to fix labels 

for d in ['preprocessed_data_objective', 'preprocessed_data_combined']:
    X_train = np.load(f'{d}/X_train.npy')
    X_val   = np.load(f'{d}/X_val.npy')
    y_train = np.load(f'{d}/y_train.npy')
    y_val   = np.load(f'{d}/y_val.npy')
    print(d, "X_train.shape:", X_train.shape, "X_val.shape:", X_val.shape)
    print("y_train pos rate:", y_train.mean(), "y_val pos rate:", y_val.mean())
    print("y_train uniques:", np.unique(y_train, return_counts=True))
    print('-'*40)