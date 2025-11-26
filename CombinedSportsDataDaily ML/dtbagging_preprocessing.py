'''
dt with bagging model from authors: 
Kayal Padmanandam, Talari Akhila, Aesuri Divya Sri, Kalekera Sunidhi, Bolle Amulya 
'''
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

class DTBaggingDataPreprocessor:
     
    """
    Preprocessing script for Decision Tree and Bagging injury prediction models
    Saves processed data to preprocessing_dtbagging folder
    """
    
    def __init__(self, output_dir='preprocessing_dtbagging'):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    def preprocess_data(self, df, target_col='injury', augmentation_factor=5):
        """
        Preprocess the runner dataset following the paper's methodology
        
        Parameters:
        -----------
        df : DataFrame
            Raw dataset with weekly runner metrics
        target_col : str
            Name of the target column (injury/non-injury)
        augmentation_factor : int
            Number of times to augment minority class (1-5)
            Paper showed best results with 5x augmentation:
            - Bagging: 94% accuracy, 91% precision, 95% F1
            - Decision Tree: 92% accuracy, 86% precision, 92% F1
        
        Returns:
        --------
        X_scaled : DataFrame
            Preprocessed and scaled features
        y_balanced : Series
            Balanced target variable
        """
        print("=" * 60)
        print("Decision Tree/Bagging Data Preprocessing")
        print("=" * 60)
        print(f"Original dataset shape: {df.shape}")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Identify minority and majority classes
        injury_cases = df[df[target_col] == 1]
        non_injury_cases = df[df[target_col] == 0]
        
        print(f"\nClass Distribution:")
        print(f"  Injury cases: {len(injury_cases)}")
        print(f"  Non-injury cases: {len(non_injury_cases)}")
        print(f"  Imbalance ratio: 1:{len(non_injury_cases)/len(injury_cases):.1f}")
        
        # Augment minority class (injury cases)
        # Paper methodology: adds 575 instances per augmentation round
        print(f"\nAugmenting minority class {augmentation_factor}x...")
        augmented_injury = injury_cases.copy()
        
        for i in range(augmentation_factor - 1):
            # Create synthetic samples with small variations
            noise_factor = 0.05
            synthetic = injury_cases.copy()
            
            # Add small random noise to numerical columns
            for col in X.columns:
                if synthetic[col].dtype in ['float64', 'int64']:
                    noise = np.random.normal(0, noise_factor * synthetic[col].std(), len(synthetic))
                    synthetic[col] = synthetic[col] + noise
            
            augmented_injury = pd.concat([augmented_injury, synthetic], ignore_index=True)
        
        # Downsample majority class to match augmented minority
        target_size = len(augmented_injury)
        downsampled_non_injury = non_injury_cases.sample(
            n=min(target_size, len(non_injury_cases)), 
            random_state=42
        )
        
        # Combine balanced dataset
        balanced_df = pd.concat([augmented_injury, downsampled_non_injury], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nBalanced Dataset:")
        print(f"  Total samples: {balanced_df.shape[0]}")
        print(f"  Injury cases: {len(augmented_injury)}")
        print(f"  Non-injury cases: {len(downsampled_non_injury)}")
        print(f"  Balance ratio: 1:1")
        
        # Split features and target
        X_balanced = balanced_df.drop(columns=[target_col])
        y_balanced = balanced_df[target_col]
        
        # Standardize features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X_balanced)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        return X_scaled, y_balanced
    
    def save_processed_data(self, X, y, train_size=0.8, val_size=0.0):
        """
        Save preprocessed data and scaler to files
        Splits data into train/test sets (no validation needed for DT/Bagging)
        
        Parameters:
        -----------
        X : DataFrame
            Preprocessed features
        y : Series
            Target variable
        train_size : float
            Proportion of data for training (default 0.8)
        val_size : float
            Not used for DT/Bagging (kept for consistency)
        """
        from sklearn.model_selection import train_test_split
        
        print("\n" + "=" * 60)
        print("Saving Processed Data")
        print("=" * 60)
        
        # Calculate test size
        test_size = 1 - train_size
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nData Split:")
        print(f"  Training set: {X_train.shape[0]} samples ({train_size:.0%})")
        print(f"  Test set: {X_test.shape[0]} samples ({test_size:.0%})")
        
        # Save datasets
        print(f"\nSaving to {self.output_dir}/...")
        
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        
        y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False, header=True)
        y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False, header=True)
        
        # Save scaler for future use
        with open(f'{self.output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature names
        with open(f'{self.output_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(self.feature_names))
        
        # Save preprocessing info
        info = {
            'total_samples': len(X),
            'n_features': len(self.feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'augmentation_factor': 5,
            'feature_names': ', '.join(self.feature_names)
        }
        
        pd.DataFrame([info]).to_csv(f'{self.output_dir}/preprocessing_info.csv', index=False)
        
        print(f"\n✓ Saved files:")
        print(f"  - X_train.csv, X_test.csv")
        print(f"  - y_train.csv, y_test.csv")
        print(f"  - scaler.pkl")
        print(f"  - feature_names.txt")
        print(f"  - preprocessing_info.csv")
        
        print("\nExpected Performance (from paper with 5x augmentation):")
        print("  Bagging:       Accuracy=0.94, Precision=0.91, F1=0.95")
        print("  Decision Tree: Accuracy=0.92, Precision=0.86, F1=0.92")
        
        return X_train, X_test, y_train, y_test


# Example usage
if __name__ == "__main__":
    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    print("Creating synthetic runner dataset...")
    features = {
        'total_sessions': np.random.randint(3, 8, n_samples),
        'rest_days': np.random.randint(0, 3, n_samples),
        'total_distance': np.random.uniform(30, 120, n_samples),
        'max_distance': np.random.uniform(10, 30, n_samples),
        'total_km_z3_z5': np.random.uniform(5, 40, n_samples),
        'tough_sessions': np.random.randint(0, 4, n_samples),
        'interval_session_days': np.random.randint(1, 5, n_samples),
        'total_km_z3_z4': np.random.uniform(5, 30, n_samples),
        'max_z3_z4_distance': np.random.uniform(5, 20, n_samples),
        'total_km_z5_t1_t2': np.random.uniform(0, 15, n_samples),
        'cross_training_hours': np.random.uniform(0, 5, n_samples),
        'strength_sessions': np.random.randint(0, 3, n_samples),
        'avg_exertion': np.random.uniform(5, 9, n_samples),
        'min_exertion': np.random.uniform(3, 7, n_samples),
        'max_exertion': np.random.uniform(7, 10, n_samples),
        'avg_training_success': np.random.uniform(5, 9, n_samples),
        'avg_recovery': np.random.uniform(4, 9, n_samples),
        'min_recovery': np.random.uniform(2, 6, n_samples),
        'max_recovery': np.random.uniform(7, 10, n_samples),
        'injury': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(features)
    
    # Initialize preprocessor
    preprocessor = DTBaggingDataPreprocessor(output_dir='preprocessing_dtbagging')
    
    # Preprocess data (5x augmentation for best results per paper)
    X, y = preprocessor.preprocess_data(df, target_col='injury', augmentation_factor=5)
    
    # Save processed data
    X_train, X_test, y_train, y_test = preprocessor.save_processed_data(X, y)
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"\nYou can now run the Decision Tree/Bagging model training scripts.")
    print(f"All data is saved in: preprocessing_dtbagging/")
    
    # To use with your own data, replace the synthetic data with:
    # df = pd.read_csv('your_runner_data.csv')
