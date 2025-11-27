'''
xg boost model based from authors: 
Kayal Padmanandam, Talari Akhila, Aesuri Divya Sri, Kalekera Sunidhi, Bolle Amulya 
'''
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class XGBoostInjuryModel:
    """
    XGBoost model for athletic runner injury prediction
    Loads preprocessed data from preprocessing_xgboost folder
    Works with V2 preprocessing scripts for timeseries (daily).csv
    """
    
    def __init__(self, data_dir='preprocessing_xgboost'):
        self.data_dir = data_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_preprocessed_data(self):
        """
        Load preprocessed data from the preprocessing folder
        """
        print("=" * 60)
        print("Loading Preprocessed Data")
        print("=" * 60)
        
        # Check if directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Directory '{self.data_dir}' not found. "
                "Please run the XGBoost preprocessing script (V2) first."
            )
        
        # Load datasets
        print(f"Loading from {self.data_dir}/...")
        
        X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        X_val = pd.read_csv(f'{self.data_dir}/X_val.csv')
        X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        
        y_train = pd.read_csv(f'{self.data_dir}/y_train.csv').squeeze()
        y_val = pd.read_csv(f'{self.data_dir}/y_val.csv').squeeze()
        y_test = pd.read_csv(f'{self.data_dir}/y_test.csv').squeeze()
        
        # Load scaler
        with open(f'{self.data_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(f'{self.data_dir}/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(self.feature_names)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, 
              n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Train XGBoost model
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training labels
        X_val : DataFrame
            Validation features
        y_val : Series
            Validation labels
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate (eta)
        """
        print("\n" + "=" * 60)
        print("Training XGBoost Model")
        print("=" * 60)
        
        # XGBoost parameters
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        
        print(f"\nModel Parameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  subsample: 0.8")
        print(f"  colsample_bytree: 0.8")
        
        # Train model
        print(f"\nTraining...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        print("✓ Training complete!")
        
    def evaluate(self, X_test, y_test, show_plots=True):
        """
        Evaluate model performance on test set
        """
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n=== XGBoost Performance ===")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")
        
        print("\nPaper's XGBoost Results (5x augmentation, 2875x2875):")
        print("  Accuracy: 0.71, Precision: 0.68, F1: 0.74")
        
        if show_plots:
            self._plot_results(cm, y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def _plot_results(self, cm, y_test, y_pred):
        """
        Plot confusion matrix and feature importance
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('XGBoost Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        
        # Get feature names for top features
        top_feature_names = [self.feature_names[i] for i in indices]
        
        axes[1].barh(range(len(indices)), importances[indices])
        axes[1].set_yticks(range(len(indices)))
        axes[1].set_yticklabels(top_feature_names)
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_title('Top 10 Most Important Features')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/xgboost_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved to: {self.data_dir}/xgboost_results.png")
        plt.show()
    
    def save_model(self, filename='xgboost_injury_model.pkl'):
        """
        Save trained model to file
        """
        filepath = f'{self.data_dir}/{filename}'
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\n✓ Model saved to: {filepath}")
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new runner data
        
        Parameters:
        -----------
        new_data : DataFrame
            New runner metrics (must have same features as training data)
        
        Returns:
        --------
        predictions : array
            Injury predictions (0=no injury, 1=injury)
        probabilities : array
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        return predictions, probabilities


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost Injury Prediction Model V2")
    print("For timeseries (daily).csv dataset")
    print("Based on: Athletic Runner Injury Prediction System (2024)")
    print("=" * 60)
    
    # Initialize model
    model = XGBoostInjuryModel(data_dir='preprocessing_xgboost')
    
    try:
        # Load preprocessed data
        X_train, X_val, X_test, y_train, y_val, y_test = model.load_preprocessed_data()
        
        # Train model
        model.train(
            X_train, y_train, 
            X_val, y_val,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        # Evaluate model
        results = model.evaluate(X_test, y_test, show_plots=True)
        
        # Save model
        model.save_model()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("\nModel Performance Summary:")
        print(f"  Accuracy:  {results['accuracy']:.2%}")
        print(f"  Precision: {results['precision']:.2%}")
        print(f"  F1 Score:  {results['f1_score']:.2%}")
        print("\nTo make predictions on new data:")
        print("  predictions, probs = model.predict_new_data(new_runner_data)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the XGBoost preprocessing script (V2) first:")
        print("  python xgboost_preprocessing_v2.py")