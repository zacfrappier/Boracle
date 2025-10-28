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

# Add Focal Loss as an option
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Custom Dataset class
class RunnerInjuryDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to torch FloatTensors
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# GRU Model (Standard setup without final Sigmoid, using BCEWithLogitsLoss)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # Dropout should only be applied if num_layers > 1
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1) # Output a single logit (no sigmoid)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        gru_out, _ = self.gru(x)
        # Use only the output from the last time step
        out = self.fc(gru_out[:, -1, :])
        # Squeeze output to (batch_size,) for BCEWithLogitsLoss
        return out.squeeze()

# Training and evaluation functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # Apply sigmoid for predictions, but not for loss calculation
        predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        targets.extend(y_batch.cpu().numpy())
    
    return total_loss / len(train_loader), predictions, targets

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            # Apply sigmoid for predictions
            predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            targets.extend(y_batch.cpu().numpy())
    
    return total_loss / len(val_loader), predictions, targets

def calculate_metrics(y_true, y_pred, threshold=CLASSIFICATION_THRESHOLD):
    y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
    
    # Check for empty confusion matrix
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_binary)) < 2:
        # Handle cases where all true labels or all predictions are the same
        tn, fp, fn, tp = (0, 0, 0, 0)
        if len(y_true) > 0:
            if y_true.sum() == 0: # All negatives
                tn = len(y_true)
            elif y_true.sum() == len(y_true): # All positives
                tp = len(y_true)
        
        # Recalculate based on predictions for better metric handling
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    else:
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate (Recall/Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate (Specificity)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 score is 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    return {
        'tpr': tpr,
        'tnr': tnr,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': cm
    }

# Visualization functions
def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, model_name):
    epochs = range(1, len(train_losses) + 1)
    
    # 1. Losses
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name}: Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss_history.png')
    plt.close()
    
    # 2. TPR/TNR
    train_tpr = [m['tpr'] for m in train_metrics]
    val_tpr = [m['tpr'] for m in val_metrics]
    train_tnr = [m['tnr'] for m in train_metrics]
    val_tnr = [m['tnr'] for m in val_metrics]
    
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, val_tpr, 'r-', label='Val TPR (Recall)')
    plt.plot(epochs, val_tnr, 'r--', label='Val TNR (Specificity)')
    plt.title(f'{model_name}: Validation TPR and TNR')
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_tpr_tnr_history.png')
    plt.close()
    
    # 3. F1 and Precision
    train_f1 = [m['f1'] for m in train_metrics]
    val_f1 = [m['f1'] for m in val_metrics]
    val_prec = [m['precision'] for m in val_metrics]
    
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, val_f1, 'r-', label='Validation F1')
    plt.plot(epochs, val_prec, 'r--', label='Validation Precision')
    plt.title(f'{model_name}: Validation F1 Score and Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_f1_prec_history.png')
    plt.close()

def plot_final_metrics(val_targets, val_pred, cm, model_name):
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=['No Injury (0)', 'Injury (1)'],
                yticklabels=['No Injury (0)', 'Injury (1)'])
    plt.title(f'{model_name}: Confusion Matrix on Validation Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(val_targets, val_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name}: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(val_targets, val_pred)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name}: Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'{model_name}_pr_curve.png')
    plt.close()
    
    return roc_auc, pr_auc

# Master training function
def run_full_training(model_type, data_dir, feature_file_name):
    print("="*60)
    print(f"--- Starting Training for {model_type} Model ---")
    print("="*60)

    # 1. Load preprocessed data
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
        
        with open(os.path.join(data_dir, feature_file_name), 'rb') as f:
            feature_names = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error loading files for {model_type}. Ensure files exist in '{data_dir}/'.")
        print(f"Missing file: {e.filename}")
        return

    # 2. Data inspection and class weights
    N_FEATURES = X_train.shape[2]
    # Calculate class weights for handling imbalance: Weight_Positive = (N_Negative / N_Positive)
    pos_weight_value = (1 - y_train.mean()) / y_train.mean()
    pos_weight = torch.tensor(pos_weight_value).to(device)

    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"\nClass distribution in training set:")
    print(f"Injury rate: {y_train.mean():.2%}")
    print(f"\nCalculated positive class weight: {pos_weight.item():.2f}")
    print(f"Number of Features (N_FEATURES): {N_FEATURES}")

    # 3. Setup Loss Function, Model, and Optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model = GRUModel(
        input_size=N_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nModel Configuration:")
    print(f"Hidden Size: {HIDDEN_SIZE}, Layers: {NUM_LAYERS}, Dropout: {DROPOUT}")
    print(f"Loss Function: {'FocalLoss' if USE_FOCAL_LOSS else 'BCEWithLogitsLoss'}")

    # 4. Create data loaders
    train_dataset = RunnerInjuryDataset(X_train, y_train)
    val_dataset = RunnerInjuryDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    
    model_save_path = f'best_gru_model_{model_type.lower()}.pt'

    print("\n--- Starting Training ---")
    for epoch in tqdm(range(NUM_EPOCHS), desc=f"Training {model_type}"):
        # Training
        train_loss, train_pred, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics_dict = calculate_metrics(train_targets, train_pred)
        
        # Validation
        val_loss, val_pred, val_targets = evaluate(
            model, val_loader, criterion, device
        )
        val_metrics_dict = calculate_metrics(val_targets, val_pred)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(train_metrics_dict)
        val_metrics.append(val_metrics_dict)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or patience_counter == EARLY_STOPPING_PATIENCE - 1:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Val F1: {val_metrics_dict['f1']:.4f}")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    # 6. Final Evaluation and Visualization
    print("\n--- Final Evaluation and Results ---")
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    val_loss, val_pred, val_targets = evaluate(model, val_loader, criterion, device)
    final_metrics = calculate_metrics(val_targets, val_pred)
    
    # Plotting
    plot_training_history(train_losses, val_losses, train_metrics, val_metrics, model_type)
    roc_auc, pr_auc = plot_final_metrics(val_targets, val_pred, final_metrics['confusion_matrix'], model_type)

    # Print final metrics
    print(f"\nFinal Validation Metrics for {model_type} Model:")
    print(f"Loss: {val_loss:.4f}")
    print(f"True Positive Rate (Recall/Sensitivity): {final_metrics['tpr']:.4f}")
    print(f"True Negative Rate (Specificity): {final_metrics['tnr']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")

    # 7. Save results
    results = {
        'model_type': model_type,
        'hyperparameters': {
            'sequence_length': SEQUENCE_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'classification_threshold': CLASSIFICATION_THRESHOLD,
            'positive_class_weight': pos_weight.item(),
            'loss_function': 'FocalLoss' if USE_FOCAL_LOSS else 'BCEWithLogitsLoss'
        },
        'final_metrics': {
            'loss': val_loss,
            'tpr': final_metrics['tpr'],
            'tnr': final_metrics['tnr'],
            'precision': final_metrics['precision'],
            'f1': final_metrics['f1'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        },
        'feature_names': feature_names
    }
    
    results_file = f'{model_type.lower()}_gru_model_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Model results saved to '{results_file}'")
    print("="*60 + "\n")

# --- Execute Training for Both Models ---
# =============================================================================

# 1. Train Objective Model
run_full_training(
    model_type='Objective', 
    data_dir='preprocessed_data_objective', 
    feature_file_name='objective_features.pkl'
)

# 2. Train Combined Model
run_full_training(
    model_type='Combined', 
    data_dir='preprocessed_data_combined', 
    feature_file_name='combined_features.pkl'
)