"""Data loading utilities for Credit Card and IEEE-CIS datasets."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class CreditCardLoader:
    """Loader for Credit Card Fraud dataset."""
    
    def __init__(self):
        self.config = config
        self.data_path = self.config.raw_data_path / self.config.get('data.creditcard.file')
    
    def load(self) -> pd.DataFrame:
        """Load Credit Card dataset."""
        print(f"Loading Credit Card dataset from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Run 'python scripts/setup_data.py' first."
            )
        
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df):,} transactions")
        print(f"  - Features: {len(df.columns)} columns")
        print(f"  - Fraud rate: {df['Class'].mean():.2%}")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Get dataset statistics."""
        return {
            'n_samples': len(df),
            'n_features': len(df.columns) - 2,  # Exclude Time, Class
            'n_fraud': df['Class'].sum(),
            'fraud_rate': df['Class'].mean(),
            'missing_values': df.isnull().sum().sum(),
            'time_span_hours': (df['Time'].max() - df['Time'].min()) / 3600,
        }


class IEEECISLoader:
    """Loader for IEEE-CIS Fraud Detection dataset."""
    
    def __init__(self):
        self.config = config
        self.raw_path = self.config.raw_data_path
        self.files = self.config.get('data.ieee_cis.files')
    
    def load_transaction(self, train: bool = True) -> pd.DataFrame:
        """Load transaction data."""
        file_key = 'train_transaction' if train else 'test_transaction'
        file_path = self.raw_path / self.files[file_key]
        
        print(f"Loading {'train' if train else 'test'} transaction data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                "Run 'python scripts/setup_data.py' first."
            )
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df):,} transactions with {len(df.columns)} columns")
        
        return df
    
    def load_identity(self, train: bool = True) -> pd.DataFrame:
        """Load identity data."""
        file_key = 'train_identity' if train else 'test_identity'
        file_path = self.raw_path / self.files[file_key]
        
        print(f"Loading {'train' if train else 'test'} identity data from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                "Run 'python scripts/setup_data.py' first."
            )
        
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df):,} identity records with {len(df.columns)} columns")
        
        return df
    
    def load_full(self, train: bool = True) -> pd.DataFrame:
        """Load transaction + identity joined."""
        trans = self.load_transaction(train)
        ident = self.load_identity(train)
        
        print(f"Joining transaction and identity on TransactionID...")
        df = trans.merge(ident, on='TransactionID', how='left')
        
        print(f"✓ Merged dataset: {len(df):,} rows, {len(df.columns)} columns")
        if train:
            print(f"  - Fraud rate: {df['isFraud'].mean():.2%}")
        
        return df
    
    def get_v_blocks_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for V-feature blocks."""
        v_cols = [col for col in df.columns if col.startswith('V')]
        
        if not v_cols:
            return pd.DataFrame()
        
        stats = pd.DataFrame({
            'feature': v_cols,
            'missing_pct': [df[col].isnull().mean() * 100 for col in v_cols],
            'unique_values': [df[col].nunique() for col in v_cols],
            'dtype': [df[col].dtype for col in v_cols]
        })
        
        return stats.sort_values('missing_pct')
    
    def get_identity_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics for identity features."""
        id_cols = [col for col in df.columns if col.startswith('id_')]
        
        if not id_cols:
            return pd.DataFrame()
        
        stats = pd.DataFrame({
            'feature': id_cols,
            'missing_pct': [df[col].isnull().mean() * 100 for col in id_cols],
            'unique_values': [df[col].nunique() for col in id_cols],
            'dtype': [df[col].dtype for col in id_cols]
        })
        
        return stats.sort_values('missing_pct')


# Convenience functions
def load_creditcard() -> pd.DataFrame:
    """Quick load Credit Card dataset."""
    loader = CreditCardLoader()
    return loader.load()


def load_ieee_cis(train: bool = True, with_identity: bool = True) -> pd.DataFrame:
    """Quick load IEEE-CIS dataset."""
    loader = IEEECISLoader()
    if with_identity:
        return loader.load_full(train)
    else:
        return loader.load_transaction(train)