"""
Deep Lane Model for IEEE-CIS Dataset
Combines supervised (XGBoost) and unsupervised (Autoencoder) learning
for complex behavioral fraud detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

# TensorFlow/Keras for Autoencoder
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. Autoencoder will not work.")


class XGBoostFraudDetector:
    """
    XGBoost classifier for supervised fraud detection on IEEE-CIS data.
    Optimized for imbalanced data with GroupKFold cross-validation.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost fraud detector.

        Args:
            params: XGBoost parameters (optional, uses defaults if None)
        """
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'scale_pos_weight': 27.6,  # Imbalance ratio for IEEE-CIS (3.5% fraud)
            'tree_method': 'hist',
            'random_state': 42
        }
        self.model = None
        self.feature_names = None
        self.best_iteration = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        n_folds: int = 5,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50
    ) -> Dict[str, float]:
        """
        Train XGBoost with GroupKFold cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            groups: Group labels for GroupKFold (e.g., DT_M for month)
            n_folds: Number of folds for cross-validation
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X.columns.tolist()

        # GroupKFold for temporal validation
        gkf = GroupKFold(n_splits=n_folds)

        fold_scores = []
        oof_preds = np.zeros(len(X))

        print(f"\n{'='*60}")
        print(f"Training XGBoost with {n_folds}-Fold GroupKFold CV")
        print(f"{'='*60}\n")

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"Fold {fold}/{n_folds}")
            print(f"  Train size: {len(train_idx):,}, Val size: {len(val_idx):,}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

            # Train
            evals = [(dtrain, 'train'), (dval, 'val')]
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=50
            )

            # Predict on validation
            val_preds = model.predict(dval)
            oof_preds[val_idx] = val_preds

            # Calculate metrics
            val_auc = roc_auc_score(y_val, val_preds)
            fold_scores.append(val_auc)

            print(f"  Fold {fold} AUC: {val_auc:.6f}")
            print(f"  Best iteration: {model.best_iteration}\n")

        # Overall OOF score
        overall_auc = roc_auc_score(y, oof_preds)

        print(f"{'='*60}")
        print(f"Cross-Validation Results:")
        print(f"  Fold AUCs: {[f'{s:.6f}' for s in fold_scores]}")
        print(f"  Mean AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
        print(f"  Overall OOF AUC: {overall_auc:.6f}")
        print(f"{'='*60}\n")

        # Train final model on all data
        print("Training final model on all data...")
        dtrain_full = xgb.DMatrix(X, label=y, feature_names=self.feature_names)
        self.model = xgb.train(
            self.params,
            dtrain_full,
            num_boost_round=int(np.mean([model.best_iteration for _ in range(n_folds)])),
            verbose_eval=False
        )
        self.best_iteration = self.model.best_iteration

        return {
            'fold_aucs': fold_scores,
            'mean_auc': np.mean(fold_scores),
            'std_auc': np.std(fold_scores),
            'oof_auc': overall_auc,
            'oof_predictions': oof_preds
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fraud probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        importance = self.model.get_score(importance_type=importance_type)
        df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(str(path / 'xgboost_model.json'))

        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'best_iteration': self.best_iteration,
            'params': self.params
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ XGBoost model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)

        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(path / 'xgboost_model.json'))

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.best_iteration = metadata['best_iteration']
        self.params = metadata['params']

        print(f"✓ XGBoost model loaded from {path}")


class AutoencoderAnomalyDetector:
    """
    Autoencoder for unsupervised anomaly detection on V-features.
    Trained on normal (non-fraud) transactions to detect anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = [128, 64, 32],
        activation: str = 'relu',
        output_activation: str = 'sigmoid'
    ):
        """
        Initialize Autoencoder.

        Args:
            input_dim: Number of input features
            encoding_dims: List of encoding layer dimensions
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for Autoencoder. Install: pip install tensorflow")

        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.activation = activation
        self.output_activation = output_activation
        self.model = None
        self.threshold = None

    def build_model(self):
        """Build autoencoder architecture."""
        # Encoder
        encoder_input = keras.Input(shape=(self.input_dim,))
        x = encoder_input

        for dim in self.encoding_dims:
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Bottleneck
        encoder_output = layers.Dense(self.encoding_dims[-1], activation=self.activation, name='bottleneck')(x)

        # Decoder
        x = encoder_output
        for dim in reversed(self.encoding_dims[:-1]):
            x = layers.Dense(dim, activation=self.activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output
        decoder_output = layers.Dense(self.input_dim, activation=self.output_activation)(x)

        # Full model
        self.model = Model(encoder_input, decoder_output, name='autoencoder')
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return self.model

    def train(
        self,
        X_normal: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train autoencoder on normal (non-fraud) transactions.

        Args:
            X_normal: Normal transaction features (fraud filtered out)
            validation_split: Validation split ratio
            epochs: Training epochs
            batch_size: Batch size
            patience: Early stopping patience

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        print(f"\n{'='*60}")
        print(f"Training Autoencoder on {len(X_normal):,} normal transactions")
        print(f"Input dimension: {self.input_dim}")
        print(f"Encoding dimensions: {self.encoding_dims}")
        print(f"{'='*60}\n")

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        # Train
        history = self.model.fit(
            X_normal, X_normal,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # Calculate reconstruction error threshold (95th percentile on training data)
        reconstructions = self.model.predict(X_normal, verbose=0)
        mse = np.mean(np.power(X_normal - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)

        print(f"\n✓ Autoencoder trained successfully")
        print(f"  Reconstruction error threshold (95th percentile): {self.threshold:.6f}\n")

        return history.history

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (reconstruction error).

        Args:
            X: Feature matrix

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse

    def predict_is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly labels.

        Args:
            X: Feature matrix

        Returns:
            Binary anomaly labels (1 = anomaly, 0 = normal)
        """
        scores = self.predict_anomaly_score(X)
        return (scores > self.threshold).astype(int)

    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(str(path / 'autoencoder_model.h5'))

        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'threshold': float(self.threshold) if self.threshold is not None else None
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Autoencoder saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        path = Path(path)

        # Load Keras model
        self.model = keras.models.load_model(str(path / 'autoencoder_model.h5'))

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.input_dim = metadata['input_dim']
        self.encoding_dims = metadata['encoding_dims']
        self.activation = metadata['activation']
        self.output_activation = metadata['output_activation']
        self.threshold = metadata['threshold']

        print(f"✓ Autoencoder loaded from {path}")


class EntityRiskAggregator:
    """
    Aggregate fraud signals by entity (device, IP, email) to generate entity risk scores.
    """

    @staticmethod
    def compute_entity_risk(
        df: pd.DataFrame,
        entity_col: str,
        fraud_col: str = 'isFraud',
        anomaly_score_col: Optional[str] = None,
        min_transactions: int = 2
    ) -> pd.DataFrame:
        """
        Compute entity-level risk scores.

        Args:
            df: DataFrame with transactions
            entity_col: Column name for entity (e.g., 'DeviceInfo', 'id_30', 'P_emaildomain')
            fraud_col: Fraud label column
            anomaly_score_col: Optional anomaly score column from Autoencoder
            min_transactions: Minimum transactions to compute risk

        Returns:
            DataFrame with entity risk scores
        """
        # Group by entity
        entity_stats = df.groupby(entity_col).agg({
            fraud_col: ['count', 'sum', 'mean']
        }).reset_index()

        entity_stats.columns = [entity_col, 'transaction_count', 'fraud_count', 'fraud_rate']

        # Add anomaly score if available
        if anomaly_score_col and anomaly_score_col in df.columns:
            anomaly_agg = df.groupby(entity_col)[anomaly_score_col].agg(['mean', 'max']).reset_index()
            anomaly_agg.columns = [entity_col, 'avg_anomaly_score', 'max_anomaly_score']
            entity_stats = entity_stats.merge(anomaly_agg, on=entity_col, how='left')

        # Filter by minimum transactions
        entity_stats = entity_stats[entity_stats['transaction_count'] >= min_transactions].copy()

        # Compute composite risk score (weighted average)
        weights = {'fraud_rate': 0.6, 'avg_anomaly_score': 0.4}

        if anomaly_score_col and anomaly_score_col in df.columns:
            # Normalize scores to 0-1
            entity_stats['fraud_rate_norm'] = entity_stats['fraud_rate']
            entity_stats['anomaly_norm'] = (
                (entity_stats['avg_anomaly_score'] - entity_stats['avg_anomaly_score'].min()) /
                (entity_stats['avg_anomaly_score'].max() - entity_stats['avg_anomaly_score'].min() + 1e-8)
            )
            entity_stats['risk_score'] = (
                weights['fraud_rate'] * entity_stats['fraud_rate_norm'] +
                weights['avg_anomaly_score'] * entity_stats['anomaly_norm']
            )
        else:
            entity_stats['risk_score'] = entity_stats['fraud_rate']

        # Sort by risk score
        entity_stats = entity_stats.sort_values('risk_score', ascending=False).reset_index(drop=True)

        return entity_stats

    @staticmethod
    def generate_entity_risk_features(
        df: pd.DataFrame,
        entity_risk_df: pd.DataFrame,
        entity_col: str
    ) -> pd.DataFrame:
        """
        Join entity risk scores back to transaction data as features.

        Args:
            df: Transaction DataFrame
            entity_risk_df: Entity risk DataFrame from compute_entity_risk()
            entity_col: Entity column name

        Returns:
            DataFrame with entity risk features added
        """
        # Select risk columns
        risk_cols = [entity_col, 'risk_score', 'fraud_rate', 'transaction_count']
        if 'avg_anomaly_score' in entity_risk_df.columns:
            risk_cols.append('avg_anomaly_score')

        # Merge
        df_with_risk = df.merge(
            entity_risk_df[risk_cols],
            on=entity_col,
            how='left',
            suffixes=('', f'_{entity_col}')
        )

        # Rename columns
        rename_map = {
            'risk_score': f'{entity_col}_risk',
            'fraud_rate': f'{entity_col}_fraud_rate',
            'transaction_count': f'{entity_col}_tx_count'
        }
        if 'avg_anomaly_score' in df_with_risk.columns:
            rename_map['avg_anomaly_score'] = f'{entity_col}_anomaly'

        df_with_risk = df_with_risk.rename(columns=rename_map)

        # Fill missing with median (for entities with <min_transactions)
        for col in rename_map.values():
            if col in df_with_risk.columns:
                df_with_risk[col].fillna(df_with_risk[col].median(), inplace=True)

        return df_with_risk


if __name__ == "__main__":
    print("Deep Lane Model Module - VPBank StreamGuard")
    print("="*60)
    print("\nComponents:")
    print("  1. XGBoostFraudDetector - Supervised fraud classification")
    print("  2. AutoencoderAnomalyDetector - Unsupervised anomaly detection")
    print("  3. EntityRiskAggregator - Entity-level risk scoring")
    print("\nUsage: Import this module in training scripts")
    print("="*60)
