import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
from tqdm.auto import tqdm
import pickle
import json
import os

#      --- Overview ---
#================================
# 1) Data Loading and Preparation
# 2) Class Imbalance handling
# 3) GRU Model Architecture
# 4) Training with GPU acceleration
# 5) Comprehensive evaluation metrics 
# 6) Performance visualization


# --- Global Configuration --- 
#==============================================================================
torch.manual_seed(9)
np.random.seed(9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Configuration Switch ---
# Set to True for full training, False for quick testing
QUICK_TEST_MODE = False

if QUICK_TEST_MODE:
    print("--- Using QUICK_TEST_MODE Hyperparameters ---")
    # Quick Test Hyperparameters (fewer epochs, smaller size)
    HIDDEN_SIZE = 16
    BATCH_SIZE = 64
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.01
else:
    print("--- Using FULL_TRAINING_MODE Hyperparameters ---")
    # Full Training Hyperparameters (your current set)
    HIDDEN_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001
    
# --- Common/Remaining Hyperparameters ---
# These are the same regardless of the mode
SEQUENCE_LENGTH = 7
NUM_LAYERS = 2
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 10
CLASSIFICATION_THRESHOLD = 0.5
USE_FOCAL_LOSS = False

# --- Utility Classes and Functions ---
#=======================================================================
