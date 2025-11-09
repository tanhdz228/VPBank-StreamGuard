"""Credit Card Fraud preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import pickle


class CreditCardPreprocessor:
    """Preprocessing pipeline for Credit Card Fraud dataset."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from Time column."""
        df = df.copy()
        
        # Hour of transaction (0-47 for 48 hours in dataset)
        df['Hour'] = (df['Time'] / 3600).astype(int)
        
        # Time period (morning, afternoon, evening, night)
        df['Time_Period'] = pd.cut(df['Hour'] % 24, 
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                     include_lowest=True)
        df['Time_Period'] = df['Time_Period'].cat.codes
        
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features."""
        df = df.copy()
        
        # Log transform amount (add 1 to avoid log(0))
        df['Amount_Log'] = np.log1p(df['Amount'])
        
        # Amount bins
        df['Amount_Bin'] = pd.qcut(df['Amount'], q=10, labels=False, duplicates='drop')
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for modeling.
        
        Args:
            df: Input dataframe with raw features
            fit: If True, fit scaler on data. If False, use existing scaler.
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        df = df.copy()
        
        # Create engineered features
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        
        # Target
        y = df['Class'].values
        
        # Feature columns
        if self.feature_columns is None:
            # Use V1-V28, Time, Amount, and engineered features
            v_cols = [f'V{i}' for i in range(1, 29)]
            self.feature_columns = v_cols + ['Time', 'Amount', 'Hour', 'Time_Period', 
                                             'Amount_Log', 'Amount_Bin']
        
        X = df[self.feature_columns].values
        
        # Handle any NaNs
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features (only Amount, Hour, Time need scaling; V* already scaled)
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y
    
    def train_val_test_split(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets with stratification.
        
        Returns:
            Dictionary with keys: 'train', 'val', 'test'
            Each value is tuple (X, y)
        """
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=self.test_size,
            stratify=df['Class'],
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['Class'],
            random_state=self.random_state
        )
        
        print(f"Split sizes:")
        print(f"  Train: {len(train_df):,} ({len(train_df)/len(df):.1%})")
        print(f"  Val:   {len(val_df):,} ({len(val_df)/len(df):.1%})")
        print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df):.1%})")
        print(f"\nFraud rates:")
        print(f"  Train: {train_df['Class'].mean():.4%}")
        print(f"  Val:   {val_df['Class'].mean():.4%}")
        print(f"  Test:  {test_df['Class'].mean():.4%}")
        
        # Prepare features
        X_train, y_train = self.prepare_features(train_df, fit=True)
        X_val, y_val = self.prepare_features(val_df, fit=False)
        X_test, y_test = self.prepare_features(test_df, fit=False)
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            test_size=state['test_size'],
            val_size=state['val_size'],
            random_state=state['random_state']
        )
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor