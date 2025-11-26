'''
xg boost model based from authors: 
Kayal Padmanandam, Talari Akhila, Aesuri Divya Sri, Kalekera Sunidhi, Bolle Amulya 
'''
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class DTBaggingInjuryModel:
    """
    Decision Tree and Bagging models for athletic runner injury prediction
    Loads preprocessed data from preprocessing_dtbagging folder
    Works with V2 preprocessing scripts for timeseries (daily).csv
    
    Paper Results (5x augmentation, 2875x2875):
    - Bagging: 94% accuracy, 91% precision, 95% F1
    - Decision Tree: 92% accuracy, 86% precision, 92% F1
    """
    
    def __init__(self, model_type='bagging', data_dir='preprocessing_dtbagging'):
        """
        Parameters:
        -----------
        model_type : str
            'bagging' or 'dt' (decision tree)
        data_dir : str
            Directory containing preprocessed data
        """
        if model_type not in ['bagging', 'dt']:
            raise ValueError("model_type must be 'bagging' or 'dt'")
        
        self.model_type = model_type
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
                "Please run the DT/Bagging preprocessing script (V2) first."
            )
        
        # Load datasets
        print(f"Loading from {self.data_dir}/...")
        
        X_train = pd.read_csv(f'{self.data_dir}/X_train.csv')
        X_test = pd.read_csv(f'{self.data_dir}/X_test.csv')
        
        y_train = pd.read_csv(f'{self.data_dir}/y_train.csv').squeeze()
        y_test = pd.read_csv(f'{self.data_dir}/y_test.csv').squeeze()
        
        # Load scaler
        with open(f'{self.data_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(f'{self.data_dir}/feature_names.txt', 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"\n✓ Data loaded successfully!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Train Decision Tree or Bagging model
        """
        print("\n" + "=" * 60)
        print(f"Training {self.model_type.upper()} Model")
        print("=" * 60)
        
        if self.model_type == 'bagging':
            # Bagging Classifier with Decision Tree as base estimator
            print("\nModel: Bagging Classifier")
            print("Base Estimator: Decision Tree")
            
            base_estimator = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=50,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            )
            
            print("\nBagging Parameters:")
            print("  n_estimators: 50")
            print("  max_samples: 0.8")
            print("  max_features: 0.8")
            print("  bootstrap: True")
            
            print("\nBase Decision Tree Parameters:")
            print("  max_depth: 10")
            print("  min_samples_split: 5")
            print("  min_samples_leaf: 2")
            
        elif self.model_type == 'dt':
            # Standalone Decision Tree
            print("\nModel: Decision Tree Classifier")
            
            self.model = DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion='gini',
                random_state=42
            )
            
            print("\nDecision Tree Parameters:")
            print("  max_depth: 15")
            print("  min_samples_split: 5")
            print("  min_samples_leaf: 2")
            print("  criterion: gini")
        
        # Train model
        print(f"\nTraining...")
        self.model.fit(X_train, y_train)
        
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
        
        print(f"\n=== {self.model_type.upper()} Performance ===")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")
        
        if self.model_type == 'bagging':
            print("\nPaper's Bagging Results (5x augmentation, 2875x2875):")
            print("  Accuracy: 0.94, Precision: 0.91, F1: 0.95")
        else:
            print("\nPaper's Decision Tree Results (5x augmentation, 2875x2875):")
            print("  Accuracy: 0.92, Precision: 0.86, F1: 0.92")
        
        if show_plots:
            self._plot_results(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def _plot_results(self, cm):
        """
        Plot confusion matrix and feature importance
        """
        if self.model_type == 'dt':
            # For Decision Tree, show both confusion matrix and feature importance
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
            axes[0].set_title(f'{self.model_type.upper()} Confusion Matrix')
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
            
        else:
            # For Bagging, just show confusion matrix
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title(f'{self.model_type.upper()} Confusion Matrix')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/{self.model_type}_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Results plot saved to: {self.data_dir}/{self.model_type}_results.png")
        plt.show()
    
    def get_feature_importance(self):
        """
        Get feature importance (Decision Tree only)
        """
        if self.model_type == 'dt' and self.model is not None:
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\n" + "=" * 60)
            print("Feature Importance Analysis")
            print("=" * 60)
            print("\nTop 10 Most Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))
            
            return feature_importance_df
        else:
            print("\nFeature importance only available for Decision Tree model")
            return None
    
    def save_model(self, filename=None):
        """
        Save trained model to file
        """
        if filename is None:
            filename = f'{self.model_type}_injury_model.pkl'
        
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
    print("Decision Tree/Bagging Injury Prediction Models V2")
    print("For timeseries (daily).csv dataset")
    print("Based on: Athletic Runner Injury Prediction System (2024)")
    print("=" * 60)
    
    # Train both models
    for model_type in ['bagging', 'dt']:
        print(f"\n{'#' * 60}")
        print(f"# Training {model_type.upper()} Model")
        print(f"{'#' * 60}\n")
        
        try:
            # Initialize model
            model = DTBaggingInjuryModel(
                model_type=model_type, 
                data_dir='preprocessing_dtbagging'
            )
            
            # Load preprocessed data
            X_train, X_test, y_train, y_test = model.load_preprocessed_data()
            
            # Train model
            model.train(X_train, y_train)
            
            # Evaluate model
            results = model.evaluate(X_test, y_test, show_plots=True)
            
            # Get feature importance (Decision Tree only)
            if model_type == 'dt':
                feature_importance = model.get_feature_importance()
            
            # Save model
            model.save_model()
            
            print(f"\n{'=' * 60}")
            print(f"{model_type.upper()} Model Training Complete!")
            print(f"{'=' * 60}")
            print(f"\nModel Performance Summary:")
            print(f"  Accuracy:  {results['accuracy']:.2%}")
            print(f"  Precision: {results['precision']:.2%}")
            print(f"  F1 Score:  {results['f1_score']:.2%}")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease run the DT/Bagging preprocessing script (V2) first:")
            print("  python dtbagging_preprocessing_v2.py")
            break
    
    print("\n" + "=" * 60)
    print("All Models Trained Successfully!")
    print("=" * 60)
    print("\nBest Performing Model (from paper): Bagging")
    print("  Accuracy: 0.94, Precision: 0.91, F1: 0.95")
    print("\nTo make predictions on new data:")
    print("  predictions, probs = model.predict_new_data(new_runner_data)")