"""
Train Deep Lane Models (XGBoost + Autoencoder) on IEEE-CIS Dataset
Generates entity risk scores for Feature Store integration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from src.data.data_loader import IEEECISLoader
from src.data.ieee_preprocessor import IEEECISPreprocessor
from src.models.deep_lane_model import (
    XGBoostFraudDetector,
    AutoencoderAnomalyDetector,
    EntityRiskAggregator
)

# Configuration
SAMPLE_SIZE = 100000  # Use 100K for development, None for full dataset
RANDOM_STATE = 42
N_FOLDS = 5

# Paths
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = Path(f'models/deep_lane_{TIMESTAMP}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data(sample_size=None):
    """Load and preprocess IEEE-CIS data."""
    print("\n" + "="*80)
    print("STEP 1: Loading IEEE-CIS Dataset")
    print("="*80 + "\n")

    # Load data
    loader = IEEECISLoader()
    df = loader.load_full(train=True)

    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")

    # Sample for development
    if sample_size and sample_size < len(df):
        print(f"\nSampling {sample_size:,} transactions for development...")
        # Stratified sampling to maintain fraud ratio
        fraud_df = df[df['isFraud'] == 1].sample(
            n=min(int(sample_size * df['isFraud'].mean()), df['isFraud'].sum()),
            random_state=RANDOM_STATE
        )
        normal_df = df[df['isFraud'] == 0].sample(
            n=sample_size - len(fraud_df),
            random_state=RANDOM_STATE
        )
        df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print(f"Sample size: {len(df):,}")
        print(f"Sample fraud rate: {df['isFraud'].mean():.4%}")

    # Preprocess
    print("\n" + "="*80)
    print("STEP 2: Preprocessing Data (Research-Backed Strategy)")
    print("="*80 + "\n")

    print("Preprocessing configuration (based on IEEE-CIS analysis):")
    print("  • V-blocks: V95-V137 (0.05% missing) + V279-V321 (0.002% missing)")
    print("  • Identity: id_23 (IP proxy), id_30 (OS), id_31 (Browser), id_33 (Screen)")
    print("  • D-column normalization by TransactionDT (key insight from 0.96 AUC notebook)")
    print("  • Feature reduction: 394 → ~150 features (optimal signal-to-noise)\n")

    preprocessor = IEEECISPreprocessor()
    df_processed = preprocessor.preprocess(df, fit=True)

    print(f"\nProcessed features: {len(df_processed.columns):,}")
    print(f"Memory usage: {df_processed.memory_usage().sum() / 1024**2:.2f} MB")

    # Show feature breakdown
    feature_groups = {
        'V-features': len([c for c in df_processed.columns if c.startswith('V')]),
        'Identity': len([c for c in df_processed.columns if c.startswith('id_')]),
        'Card': len([c for c in df_processed.columns if c.startswith('card')]),
        'C-features': len([c for c in df_processed.columns if c.startswith('C')]),
        'D-features': len([c for c in df_processed.columns if c.startswith('D')]),
        'M-features': len([c for c in df_processed.columns if c.startswith('M')]),
        'Temporal': len([c for c in df_processed.columns if c.startswith('DT_')]),
        'Other': len(df_processed.columns) - sum([
            len([c for c in df_processed.columns if c.startswith(p)])
            for p in ['V', 'id_', 'card', 'C', 'D', 'M', 'DT_']
        ])
    }
    print("\nFeature groups breakdown:")
    for group, count in feature_groups.items():
        print(f"  {group}: {count}")

    # Save preprocessor
    preprocessor_path = OUTPUT_DIR / 'preprocessor.pkl'
    preprocessor.save(preprocessor_path)
    print(f"\n✓ Preprocessor saved to {preprocessor_path}")

    # Memory cleanup
    gc.collect()

    # Prepare for training
    X, y, feature_names = preprocessor.prepare_for_training(df_processed)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Number of features: {len(feature_names)}")

    # Extract groups for GroupKFold (use DT_M if available)
    if 'DT_M' in df_processed.columns:
        groups = df_processed['DT_M'].values
        print(f"Using DT_M (month) for GroupKFold: {len(np.unique(groups))} unique months")
    else:
        groups = None
        print("Warning: DT_M not found, will use standard KFold")

    return X, y, groups, feature_names, df_processed, preprocessor


def train_xgboost(X, y, groups, feature_names):
    """Train XGBoost fraud detector."""
    print("\n" + "="*80)
    print("STEP 3: Training XGBoost Fraud Detector")
    print("="*80 + "\n")

    # Initialize model
    xgb_detector = XGBoostFraudDetector()

    # Train with GroupKFold
    if groups is not None:
        results = xgb_detector.train(
            X=pd.DataFrame(X, columns=feature_names),
            y=pd.Series(y),
            groups=pd.Series(groups),
            n_folds=N_FOLDS,
            num_boost_round=1000,
            early_stopping_rounds=50
        )
    else:
        # Fallback to regular training if no groups
        print("Warning: Training without GroupKFold (groups not provided)")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            xgb_detector.params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        xgb_detector.model = model
        xgb_detector.feature_names = feature_names

        from sklearn.metrics import roc_auc_score
        val_preds = model.predict(dval)
        val_auc = roc_auc_score(y_val, val_preds)

        results = {
            'val_auc': val_auc,
            'best_iteration': model.best_iteration
        }

    # Save model
    xgb_path = OUTPUT_DIR / 'xgboost'
    xgb_detector.save(xgb_path)

    # Feature importance
    feature_importance = xgb_detector.get_feature_importance()
    feature_importance.to_csv(OUTPUT_DIR / 'xgboost_feature_importance.csv', index=False)

    print(f"\nTop 20 Features:")
    print(feature_importance.head(20).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 20 Feature Importance (XGBoost)', fontsize=14)
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'xgboost_feature_importance.png', dpi=300)
    print(f"✓ Feature importance plot saved to {OUTPUT_DIR / 'xgboost_feature_importance.png'}")
    plt.close()

    return xgb_detector, results


def train_autoencoder(X, y, feature_names):
    """Train Autoencoder on normal (non-fraud) transactions."""
    print("\n" + "="*80)
    print("STEP 4: Training Autoencoder for Anomaly Detection")
    print("="*80 + "\n")

    # Filter V-features only (for autoencoder)
    # Preprocessor already selected high-quality V-blocks: V95-V137, V279-V321
    # These have <1% missing values and highest signal quality (from IEEE analysis)
    v_features = [f for f in feature_names if f.startswith('V')]
    print(f"Using {len(v_features)} V-features for Autoencoder")
    print(f"  Note: V95-V137 (0.05% missing) + V279-V321 (0.002% missing)")
    print(f"  These are the highest-quality V-blocks from IEEE-CIS dataset\n")

    if len(v_features) == 0:
        print("Warning: No V-features found. Skipping Autoencoder training.")
        return None

    # Get V-feature indices
    v_indices = [i for i, f in enumerate(feature_names) if f.startswith('V')]
    X_v = X[:, v_indices]

    # Filter normal transactions only
    X_normal = X_v[y == 0]
    print(f"Training on {len(X_normal):,} normal transactions")

    # Initialize Autoencoder
    autoencoder = AutoencoderAnomalyDetector(
        input_dim=len(v_features),
        encoding_dims=[128, 64, 32],
        activation='relu',
        output_activation='sigmoid'
    )

    # Build and train
    autoencoder.build_model()
    print("\nAutoencoder Architecture:")
    autoencoder.model.summary()

    history = autoencoder.train(
        X_normal,
        validation_split=0.2,
        epochs=100,
        batch_size=256,
        patience=10
    )

    # Save model
    ae_path = OUTPUT_DIR / 'autoencoder'
    autoencoder.save(ae_path)

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history['loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Autoencoder Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE
    ax2.plot(history['mae'], label='Train MAE')
    ax2.plot(history['val_mae'], label='Val MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Autoencoder Training MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'autoencoder_training.png', dpi=300)
    print(f"✓ Training history plot saved to {OUTPUT_DIR / 'autoencoder_training.png'}")
    plt.close()

    # Predict anomaly scores on full dataset
    anomaly_scores = autoencoder.predict_anomaly_score(X_v)

    # Plot anomaly score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(anomaly_scores[y == 0], bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(anomaly_scores[y == 1], bins=50, alpha=0.7, label='Fraud', density=True)
    plt.axvline(autoencoder.threshold, color='red', linestyle='--', label=f'Threshold (95th percentile)')
    plt.xlabel('Anomaly Score (Reconstruction Error)')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'anomaly_score_distribution.png', dpi=300)
    print(f"✓ Anomaly score distribution plot saved to {OUTPUT_DIR / 'anomaly_score_distribution.png'}")
    plt.close()

    return autoencoder, anomaly_scores, v_features


def generate_entity_risk_scores(df_processed, anomaly_scores):
    """Generate entity risk scores."""
    print("\n" + "="*80)
    print("STEP 5: Generating Entity Risk Scores")
    print("="*80 + "\n")

    # Add anomaly scores to df
    df_with_anomaly = df_processed.copy()
    df_with_anomaly['anomaly_score'] = anomaly_scores

    # Entity columns to analyze
    entity_columns = []

    # CRITICAL: IP Proxy (id_23) - marked as CRITICAL in IEEE-CIS analysis
    if 'id_23' in df_with_anomaly.columns:
        entity_columns.append('id_23')  # IP proxy type (CRITICAL fraud signal)

    # Device-related (Priority Tier 1)
    if 'DeviceInfo' in df_with_anomaly.columns:
        entity_columns.append('DeviceInfo')
    if 'id_30' in df_with_anomaly.columns:
        entity_columns.append('id_30')  # OS version
    if 'id_31' in df_with_anomaly.columns:
        entity_columns.append('id_31')  # Browser version
    if 'id_33' in df_with_anomaly.columns:
        entity_columns.append('id_33')  # Screen resolution

    # Email domain
    if 'P_emaildomain' in df_with_anomaly.columns:
        entity_columns.append('P_emaildomain')

    # Card
    if 'card1' in df_with_anomaly.columns:
        entity_columns.append('card1')

    print(f"Analyzing {len(entity_columns)} entity types: {entity_columns}")
    print(f"  Priority: id_23 (IP proxy - CRITICAL fraud signal)")
    print(f"  Device: id_30 (OS), id_31 (Browser), id_33 (Screen)")
    print(f"  Other: DeviceInfo, P_emaildomain, card1\n")

    # Compute risk for each entity
    entity_risk_dfs = {}

    for entity_col in entity_columns:
        print(f"Computing risk for {entity_col}...")

        entity_risk = EntityRiskAggregator.compute_entity_risk(
            df_with_anomaly,
            entity_col=entity_col,
            fraud_col='isFraud',
            anomaly_score_col='anomaly_score',
            min_transactions=2
        )

        # Save
        entity_risk.to_csv(OUTPUT_DIR / f'entity_risk_{entity_col}.csv', index=False)
        entity_risk_dfs[entity_col] = entity_risk

        print(f"  Entities analyzed: {len(entity_risk):,}")
        print(f"  Top 5 risky {entity_col}:")
        print(entity_risk.head(5)[['risk_score', 'fraud_rate', 'transaction_count']].to_string(index=False))
        print()

    # Combine all entity risks
    all_entity_risks = []
    for entity_col, entity_risk in entity_risk_dfs.items():
        entity_risk['entity_type'] = entity_col
        entity_risk = entity_risk.rename(columns={entity_col: 'entity_id'})
        all_entity_risks.append(entity_risk)

    combined_entity_risk = pd.concat(all_entity_risks, ignore_index=True)
    combined_entity_risk.to_csv(OUTPUT_DIR / 'entity_risk_combined.csv', index=False)

    print(f"✓ Combined entity risk saved to {OUTPUT_DIR / 'entity_risk_combined.csv'}")
    print(f"  Total entities: {len(combined_entity_risk):,}")

    # Show overall risk distribution
    print(f"\n{'='*60}")
    print("Risk Score Distribution (across all entities):")
    print(f"  Min: {combined_entity_risk['risk_score'].min():.4f}")
    print(f"  25th percentile: {combined_entity_risk['risk_score'].quantile(0.25):.4f}")
    print(f"  Median: {combined_entity_risk['risk_score'].median():.4f}")
    print(f"  75th percentile: {combined_entity_risk['risk_score'].quantile(0.75):.4f}")
    print(f"  Max: {combined_entity_risk['risk_score'].max():.4f}")
    print(f"  High-risk entities (score > 0.5): {(combined_entity_risk['risk_score'] > 0.5).sum():,}")
    print(f"{'='*60}\n")

    return entity_risk_dfs


def save_results(xgb_results, ae_results=None):
    """Save training results summary."""
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80 + "\n")

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Convert xgb_results to serializable format
    xgb_results_serializable = convert_to_serializable(xgb_results)

    results = {
        'timestamp': TIMESTAMP,
        'dataset': 'IEEE-CIS',
        'sample_size': SAMPLE_SIZE if SAMPLE_SIZE else 'FULL',
        'xgboost': xgb_results_serializable,
        'autoencoder': {
            'trained': ae_results is not None,
            'threshold': float(ae_results[0].threshold) if ae_results else None
        } if ae_results else None
    }

    # Save JSON
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {OUTPUT_DIR / 'results.json'}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("VPBank StreamGuard - Deep Lane Training")
    print("="*80)
    print(f"Configuration:")
    print(f"  Sample size: {SAMPLE_SIZE:,}" if SAMPLE_SIZE else "  Sample size: FULL DATASET")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")

    try:
        # Step 1: Load and preprocess
        X, y, groups, feature_names, df_processed, preprocessor = load_and_preprocess_data(SAMPLE_SIZE)

        # Step 2: Train XGBoost
        xgb_detector, xgb_results = train_xgboost(X, y, groups, feature_names)

        # Memory cleanup after XGBoost
        gc.collect()

        # Step 3: Train Autoencoder (if Keras available)
        try:
            ae_detector, anomaly_scores, v_features = train_autoencoder(X, y, feature_names)
            ae_results = (ae_detector, anomaly_scores, v_features)
        except Exception as e:
            print(f"\nWarning: Autoencoder training failed: {e}")
            print("Continuing without Autoencoder...")
            ae_results = None
            anomaly_scores = np.zeros(len(X))  # Dummy scores

        # Memory cleanup after Autoencoder
        gc.collect()

        # Step 4: Generate entity risk scores
        entity_risk_dfs = generate_entity_risk_scores(df_processed, anomaly_scores)

        # Step 5: Save results
        save_results(xgb_results, ae_results)

        # Final summary
        print("\n" + "="*80)
        print("DEEP LANE TRAINING COMPLETE! ✓")
        print("="*80)
        print(f"\nModels and artifacts saved to: {OUTPUT_DIR}")
        print("\nTraining Summary:")
        print(f"  XGBoost Mean AUC: {xgb_results.get('mean_auc', 'N/A'):.4f}")
        print(f"  XGBoost OOF AUC: {xgb_results.get('oof_auc', 'N/A'):.4f}")
        if ae_results:
            print(f"  Autoencoder threshold: {ae_results[0].threshold:.6f}")
        print(f"  Entity risk files: {len(entity_risk_dfs)} types")
        print("\nNext steps:")
        print("1. Review results.json for performance metrics")
        print("2. Check entity_risk_*.csv for entity risk scores")
        print("3. Proceed to Feature Store integration (Day 6-7)")
        print("\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial results may be in: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n\nERROR: Training failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be in: {OUTPUT_DIR}")
        raise


if __name__ == "__main__":
    main()
