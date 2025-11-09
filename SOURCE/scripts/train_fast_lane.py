"""Train Fast Lane model on Credit Card dataset - Baseline."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json

from src.data.data_loader import load_creditcard
from src.data.creditcard_preprocessor import CreditCardPreprocessor
from src.utils.config import config


def plot_roc_pr_curves(y_true, y_pred_proba, model_name, save_dir):
    """Plot ROC and PR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    ax1.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'{model_name} - ROC Curve', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    ax2.plot(recall, precision, linewidth=2)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f'{model_name.lower().replace(" ", "_")}_curves.png', dpi=150)
    print(f"✓ Saved curves to {save_dir}")
    plt.close()


def calculate_recall_at_fpr(y_true, y_pred_proba, target_fpr=0.01):
    """Calculate recall at specific FPR threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Find threshold where FPR is closest to target
    idx = np.argmin(np.abs(fpr - target_fpr))
    recall_at_target = tpr[idx]
    threshold_at_target = thresholds[idx]
    actual_fpr = fpr[idx]
    
    return {
        'recall': recall_at_target,
        'threshold': threshold_at_target,
        'actual_fpr': actual_fpr
    }


def train_logistic_baseline(X_train, y_train, X_val, y_val):
    """Train Logistic Regression baseline."""
    print("\n" + "="*60)
    print("Training Logistic Regression Baseline")
    print("="*60)
    
    model = LogisticRegression(
        C=0.01,  # Regularization
        max_iter=1000,
        solver='saga',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Val AUC:   {val_auc:.4f}")
    
    # Recall at 1% FPR
    recall_metrics = calculate_recall_at_fpr(y_val, y_val_pred, target_fpr=0.01)
    print(f"\nRecall@1%FPR: {recall_metrics['recall']:.4f}")
    print(f"Threshold:     {recall_metrics['threshold']:.4f}")
    print(f"Actual FPR:    {recall_metrics['actual_fpr']:.4f}")
    
    return model, {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'recall_at_1pct_fpr': recall_metrics['recall'],
        'threshold': recall_metrics['threshold']
    }


def train_lightgbm_baseline(X_train, y_train, X_val, y_val):
    """Train LightGBM baseline."""
    print("\n" + "="*60)
    print("Training LightGBM Baseline")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    print("Training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Predictions
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    # Metrics
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val AUC:   {val_auc:.4f}")
    
    # Recall at 1% FPR
    recall_metrics = calculate_recall_at_fpr(y_val, y_val_pred, target_fpr=0.01)
    print(f"\nRecall@1%FPR: {recall_metrics['recall']:.4f}")
    print(f"Threshold:     {recall_metrics['threshold']:.4f}")
    print(f"Actual FPR:    {recall_metrics['actual_fpr']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    return model, {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'recall_at_1pct_fpr': recall_metrics['recall'],
        'threshold': recall_metrics['threshold'],
        'best_iteration': model.best_iteration,
        'feature_importance': feature_importance
    }


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("VPBank StreamGuard - Fast Lane Training (Credit Card Baseline)")
    print("="*80 + "\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/fast_lane_baseline_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load data
    print("Loading Credit Card dataset...")
    df = load_creditcard()
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = CreditCardPreprocessor(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    splits = preprocessor.train_val_test_split(df)
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor.pkl')
    
    # Train models
    results = {}
    
    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_baseline(X_train, y_train, X_val, y_val)
    results['logistic_regression'] = lr_metrics
    
    # Save LR model
    with open(output_dir / 'logistic_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"✓ Saved Logistic model")
    
    # Plot LR curves
    y_val_pred_lr = lr_model.predict_proba(X_val)[:, 1]
    plot_roc_pr_curves(y_val, y_val_pred_lr, "Logistic Regression", output_dir)
    
    # 2. LightGBM
    lgb_model, lgb_metrics = train_lightgbm_baseline(X_train, y_train, X_val, y_val)
    results['lightgbm'] = lgb_metrics
    
    # Save LightGBM model
    lgb_model.save_model(str(output_dir / 'lightgbm_model.txt'))
    print(f"✓ Saved LightGBM model")
    
    # Save feature importance
    lgb_metrics['feature_importance'].to_csv(output_dir / 'feature_importance.csv', index=False)
    
    # Plot LGB curves
    y_val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    plot_roc_pr_curves(y_val, y_val_pred_lgb, "LightGBM", output_dir)
    
    # Test set evaluation (best model)
    print("\n" + "="*60)
    print("Test Set Evaluation (LightGBM)")
    print("="*60)
    
    y_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_recall_metrics = calculate_recall_at_fpr(y_test, y_test_pred, target_fpr=0.01)
    
    print(f"Test AUC:         {test_auc:.4f}")
    print(f"Recall@1%FPR:     {test_recall_metrics['recall']:.4f}")
    print(f"Threshold:        {test_recall_metrics['threshold']:.4f}")
    
    results['test'] = {
        'auc': test_auc,
        'recall_at_1pct_fpr': test_recall_metrics['recall'],
        'threshold': test_recall_metrics['threshold']
    }
    
    # Plot test curves
    plot_roc_pr_curves(y_test, y_test_pred, "LightGBM Test", output_dir)
    
    # Save all results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy types to native Python types
        results_serializable = {}
        for model, metrics in results.items():
            results_serializable[model] = {}
            for key, value in metrics.items():
                if key == 'feature_importance':
                    continue  # Already saved as CSV
                if isinstance(value, (np.integer, np.floating)):
                    results_serializable[model][key] = float(value)
                else:
                    results_serializable[model][key] = value
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ All results saved to {output_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"\nLogistic Regression:")
    print(f"  Val AUC:          {lr_metrics['val_auc']:.4f}")
    print(f"  Recall@1%FPR:     {lr_metrics['recall_at_1pct_fpr']:.4f}")
    print(f"\nLightGBM:")
    print(f"  Val AUC:          {lgb_metrics['val_auc']:.4f}")
    print(f"  Recall@1%FPR:     {lgb_metrics['recall_at_1pct_fpr']:.4f}")
    print(f"  Best iteration:   {lgb_metrics['best_iteration']}")
    print(f"\nTest Performance (LightGBM):")
    print(f"  Test AUC:         {test_auc:.4f}")
    print(f"  Recall@1%FPR:     {test_recall_metrics['recall']:.4f}")
    
    # Check if we met targets
    print(f"\n{'='*80}")
    print("TARGET VALIDATION")
    print("="*80)
    target_auc = 0.95
    target_recall = 0.70
    
    print(f"Target AUC:       {target_auc:.2f}")
    print(f"Achieved AUC:     {test_auc:.4f} {'✓ PASS' if test_auc >= target_auc else '✗ FAIL'}")
    print(f"\nTarget Recall@1%: {target_recall:.2f}")
    print(f"Achieved Recall:  {test_recall_metrics['recall']:.4f} {'✓ PASS' if test_recall_metrics['recall'] >= target_recall else '✗ FAIL'}")
    
    print(f"\n{'='*80}")
    print(f"✓ Fast Lane baseline training complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()