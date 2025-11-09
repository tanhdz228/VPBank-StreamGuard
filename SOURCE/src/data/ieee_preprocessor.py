"""IEEE-CIS Fraud Detection preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, List
import pickle
import gc


class IEEECISPreprocessor:
    """Preprocessing pipeline for IEEE-CIS dataset with V-blocks and Identity."""
    
    def __init__(self, v_blocks_keep: List[Tuple[int, int]] = None, 
                 identity_priority: List[str] = None,
                 random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            v_blocks_keep: List of (start, end) tuples for V features to keep
            identity_priority: List of identity feature names to prioritize
        """
        self.random_state = random_state
        
        # V-blocks to keep (default from config)
        self.v_blocks_keep = v_blocks_keep or [(95, 137), (279, 321)]
        
        # Identity features to prioritize (low missing)
        self.identity_priority = identity_priority or ['id_30', 'id_31', 'id_23', 'id_33']
        
        # Storage for encoders
        self.label_encoders = {}
        self.feature_columns = None
        
    def normalize_d_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize D columns by time (key insight from XGB notebook).
        D1-D15 represent time deltas, normalize by TransactionDT.
        """
        df = df.copy()
        
        # D columns to normalize (skip D1, D2, D3, D5, D9 as per notebook)
        d_cols_to_normalize = ['D4', 'D6', 'D7', 'D8', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']
        
        for col in d_cols_to_normalize:
            if col in df.columns:
                df[col] = df[col] - df['TransactionDT'] / np.float32(24*60*60)
        
        print(f"✓ Normalized {len([c for c in d_cols_to_normalize if c in df.columns])} D columns")
        return df
    
    def select_v_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select V features based on configured blocks.
        Reduces from 339 to ~86 features (2 blocks of 43 each).
        """
        df = df.copy()
        
        # Get all V columns
        all_v_cols = [col for col in df.columns if col.startswith('V')]
        
        # Select V columns in specified blocks
        v_cols_to_keep = []
        for start, end in self.v_blocks_keep:
            block_cols = [f'V{i}' for i in range(start, end + 1) if f'V{i}' in df.columns]
            v_cols_to_keep.extend(block_cols)
        
        # Drop V columns not in keep list
        v_cols_to_drop = [col for col in all_v_cols if col not in v_cols_to_keep]
        df = df.drop(columns=v_cols_to_drop)
        
        print(f"✓ Reduced V features from {len(all_v_cols)} to {len(v_cols_to_keep)}")
        return df
    
    def select_identity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select identity features, prioritizing low-missing features.
        Keep id_30 (OS), id_31 (Browser), id_23 (IP proxy), id_33 (Screen).
        """
        df = df.copy()
        
        all_id_cols = [col for col in df.columns if col.startswith('id_')]
        
        # Keep priority features + others with low missing rate
        id_cols_to_keep = []
        
        for col in all_id_cols:
            if col in self.identity_priority:
                id_cols_to_keep.append(col)
            else:
                # Keep if missing rate < 50%
                missing_rate = df[col].isnull().mean()
                if missing_rate < 0.5:
                    id_cols_to_keep.append(col)
        
        # Drop high-missing identity columns
        id_cols_to_drop = [col for col in all_id_cols if col not in id_cols_to_keep]
        df = df.drop(columns=id_cols_to_drop)
        
        print(f"✓ Kept {len(id_cols_to_keep)}/{len(all_id_cols)} identity features")
        return df
    
    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-based features."""
        df = df.copy()
        
        # Cents from amount (key feature from notebook)
        df['cents'] = (df['TransactionAmt'] - np.floor(df['TransactionAmt'])).astype('float32')
        
        # Amount bins
        df['TransactionAmt_Log'] = np.log1p(df['TransactionAmt'])
        
        # Time features from TransactionDT
        import datetime
        START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
        df['DT'] = df['TransactionDT'].apply(lambda x: START_DATE + datetime.timedelta(seconds=x))
        df['DT_M'] = (df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month
        df['DT_W'] = (df['DT'].dt.year - 2017) * 52 + df['DT'].dt.isocalendar().week
        df['DT_D'] = (df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear
        df['DT_hour'] = df['DT'].dt.hour
        df['DT_day_week'] = df['DT'].dt.dayofweek
        df['DT_day'] = df['DT'].dt.day
        
        # Drop datetime column
        df = df.drop(columns=['DT'])
        
        print("✓ Created transaction features")
        return df
    
    def frequency_encode(self, df: pd.DataFrame, cols: List[str], suffix: str = 'FE') -> pd.DataFrame:
        """
        Frequency encode categorical columns.
        Replace categories with their frequency in the dataset.
        """
        df = df.copy()
        
        for col in cols:
            if col in df.columns:
                # Calculate frequencies
                vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
                vc[-1] = -1  # For missing values
                
                # Map frequencies
                nm = f'{col}_{suffix}'
                df[nm] = df[col].map(vc).fillna(-1).astype('float32')
        
        return df
    
    def combine_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine features to create interaction features.
        Key combinations from XGB notebook.
        """
        df = df.copy()
        
        # card1 + addr1
        if 'card1' in df.columns and 'addr1' in df.columns:
            df['card1_addr1'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
        
        # card1 + addr1 + P_emaildomain
        if 'card1_addr1' in df.columns and 'P_emaildomain' in df.columns:
            df['card1_addr1_email'] = df['card1_addr1'] + '_' + df['P_emaildomain'].astype(str)
        
        print("✓ Created combined features")
        return df
    
    def label_encode(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Label encode categorical features.
        Store encoders for inverse transform if needed.
        """
        df = df.copy()

        # String/categorical columns
        str_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                    'DeviceType', 'DeviceInfo']

        # Add combined feature columns (created in combine_features())
        combined_cols = ['card1_addr1', 'card1_addr1_email']
        str_cols.extend([col for col in combined_cols if col in df.columns])

        # Add identity string columns that exist
        id_str_cols = [f'id_{i}' for i in [12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38]]
        str_cols.extend([col for col in id_str_cols if col in df.columns])
        
        for col in str_cols:
            if col in df.columns:
                if fit:
                    # Fit new encoder
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str).fillna('missing'))
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df[col] = df[col].astype(str).fillna('missing')
                        df[col] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[col] = -1
                
                df[col] = df[col].astype('int16')
        
        print(f"✓ Label encoded {len([c for c in str_cols if c in df.columns])} categorical features")
        return df
    
    def reduce_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage by optimizing dtypes."""
        df = df.copy()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object' and col != 'isFraud':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Raw IEEE-CIS dataframe (transaction + identity merged)
            fit: If True, fit encoders. If False, use existing encoders.
            
        Returns:
            Preprocessed dataframe ready for modeling
        """
        print(f"\n{'='*60}")
        print(f"IEEE-CIS Preprocessing Pipeline")
        print(f"Input shape: {df.shape}")
        print(f"{'='*60}\n")
        
        # Step 1: Normalize D columns
        df = self.normalize_d_columns(df)
        
        # Step 2: Select V features
        df = self.select_v_features(df)
        
        # Step 3: Select identity features
        df = self.select_identity_features(df)
        
        # Step 4: Create transaction features
        df = self.create_transaction_features(df)
        
        # Step 5: Combine features
        df = self.combine_features(df)
        
        # Step 6: Frequency encode important columns
        freq_cols = ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain']
        freq_cols = [c for c in freq_cols if c in df.columns]
        df = self.frequency_encode(df, freq_cols)
        
        # Step 7: Label encode categoricals
        df = self.label_encode(df, fit=fit)
        
        # Step 8: Fill remaining NaNs with -1
        df = df.fillna(-1)
        
        # Step 9: Reduce memory
        df = self.reduce_memory(df)
        
        print(f"\n{'='*60}")
        print(f"Preprocessing Complete")
        print(f"Output shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*60}\n")
        
        # Clean up
        gc.collect()
        
        return df
    
    def prepare_for_training(self, df: pd.DataFrame, target_col: str = 'isFraud'):
        """
        Prepare preprocessed data for training.

        Returns:
            X: Feature matrix (numpy array with numeric dtype)
            y: Target vector
            feature_names: List of feature names
        """
        # Remove columns not used for training
        exclude_cols = [target_col, 'TransactionID', 'TransactionDT']

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # CRITICAL FIX: Convert any remaining object dtypes to numeric
        # This prevents XGBoost "object dtype" error
        df_features = df[feature_cols].copy()

        # Check for object columns and convert
        object_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        if len(object_cols) > 0:
            print(f"\nWARNING: Found {len(object_cols)} object dtype columns:")
            print(f"  {object_cols}")
            print(f"Converting to numeric (strings will become NaN)...")

            for col in object_cols:
                # Try to convert to numeric, coercing errors to NaN
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

            print(f"✓ Converted all object columns to numeric dtypes")

        # Fill any NaNs that resulted from coercion
        df_features = df_features.fillna(-1)

        # Verify all columns are numeric before conversion
        remaining_object_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        if len(remaining_object_cols) > 0:
            raise ValueError(
                f"ERROR: {len(remaining_object_cols)} columns still have object dtype after conversion: {remaining_object_cols}\n"
                f"This indicates a bug in the preprocessing pipeline. "
                f"All features must be numeric before training."
            )

        # Convert to numpy array with float32 dtype (memory efficient)
        X = df_features.to_numpy(dtype=np.float32)
        y = df[target_col].values if target_col in df.columns else None

        # Final verification
        if X.dtype == object:
            raise ValueError(
                f"ERROR: Feature matrix X has object dtype after conversion!\n"
                f"Expected: float32, Got: {X.dtype}\n"
                f"This should not happen. Check for non-numeric values in features."
            )

        print(f"\n✓ Feature matrix prepared: shape={X.shape}, dtype={X.dtype}")

        self.feature_columns = feature_cols

        return X, y, feature_cols
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'v_blocks_keep': self.v_blocks_keep,
            'identity_priority': self.identity_priority,
            'random_state': self.random_state
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            v_blocks_keep=state['v_blocks_keep'],
            identity_priority=state['identity_priority'],
            random_state=state['random_state']
        )
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.feature_columns = state['feature_columns']
        
        print(f"✓ Preprocessor loaded from {filepath}")
        return preprocessor