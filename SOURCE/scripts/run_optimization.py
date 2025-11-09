"""
Run Optimization Tasks for VPBank StreamGuard

Orchestrates:
1. Threshold tuning (cost-based optimization)
2. Model performance monitoring (PSI, KS, drift detection)
3. CloudWatch metrics publishing

Usage:
    python scripts/run_optimization.py --mode all
    python scripts/run_optimization.py --mode threshold-only
    python scripts/run_optimization.py --mode drift-only
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.optimization.threshold_tuner import ThresholdTuner, BusinessMetrics, ThresholdConfig
from src.monitoring.model_performance import ModelPerformanceMonitor, DriftThresholds
from src.monitoring.cloudwatch_metrics import MetricsPublisher


class OptimizationOrchestrator:
    """Orchestrate all optimization tasks."""

    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models"):
        """
        Initialize orchestrator.

        Args:
            data_dir: Directory containing processed data
            models_dir: Directory containing trained models
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.output_dir = project_root / "optimization_results"
        self.output_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("VPBANK STREAMGUARD - OPTIMIZATION ORCHESTRATOR")
        print("="*70)
        print(f"\nData Directory: {self.data_dir}")
        print(f"Models Directory: {self.models_dir}")
        print(f"Output Directory: {self.output_dir}")

    def load_validation_data(self) -> tuple:
        """
        Load validation data for optimization.

        Returns:
            Tuple of (risk_scores, true_labels, features_df)
        """
        print("\n[INFO] Loading validation data...")

        # Try to load from Fast Lane baseline
        fast_lane_dirs = list(self.models_dir.glob("fast_lane_baseline_*"))

        if not fast_lane_dirs:
            print("[ERROR] No Fast Lane baseline found. Please run training first.")
            return None, None, None

        # Use most recent
        model_dir = sorted(fast_lane_dirs)[-1]
        print(f"[INFO] Using model from: {model_dir}")

        # Try to load preprocessor
        preprocessor_path = model_dir / "preprocessor.pkl"
        if not preprocessor_path.exists():
            print("[ERROR] Preprocessor not found. Cannot load validation data.")
            return None, None, None

        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        # Load model
        model_path = model_dir / "logistic_model.pkl"
        if not model_path.exists():
            model_path = model_dir / "lr_model.pkl"

        if not model_path.exists():
            print("[ERROR] Model file not found.")
            return None, None, None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load validation data
        # Note: In production, this would load from a dedicated validation set
        # For demo, we'll generate synthetic data similar to training

        print("[INFO] Generating synthetic validation data for optimization...")

        np.random.seed(43)  # Different seed from training
        n_samples = 5000
        # Use higher fraud rate for meaningful optimization (similar to IEEE-CIS 3.5%)
        # Credit card fraud rate (0.00172) would give only ~8 fraud samples, which is too few
        fraud_rate = 0.035  # 3.5% fraud rate (IEEE-CIS level) for robust optimization

        # Generate synthetic features (V1-V28, Time, Amount)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        # Simulate PCA features with distinguishable patterns
        # Fraud cases: More extreme values in key components (V10, V4, V14, V12, V11 are top features)
        # Make fraud features have different means and higher variance to be distinguishable
        fraud_features = np.random.randn(n_fraud, 28) * 1.5  # Higher variance

        # Shift mean for top fraud indicators (V10, V4, V14, V12, V11 = indices 9, 3, 13, 11, 10)
        fraud_features[:, 9] += 1.5   # V10 shift
        fraud_features[:, 3] += 1.2   # V4 shift
        fraud_features[:, 13] += 1.0  # V14 shift
        fraud_features[:, 11] += 0.8  # V12 shift
        fraud_features[:, 10] += 0.8  # V11 shift

        # Legitimate cases: Standard normal distribution
        legit_features = np.random.randn(n_legit, 28)

        # Add Time (0-172792 seconds = ~48 hours)
        # Fraud more likely at unusual hours (late night)
        fraud_time = np.concatenate([
            np.random.uniform(0, 21600, int(n_fraud * 0.4)),      # 40% late night (0-6am)
            np.random.uniform(21600, 172792, int(n_fraud * 0.6))  # 60% rest of day
        ])
        np.random.shuffle(fraud_time)

        # Legitimate more evenly distributed
        legit_time = np.random.uniform(0, 172792, n_legit)

        # Add Amount (log-normal distribution)
        # Fraud: Lower amounts on average (as seen in training data)
        fraud_amount = np.random.lognormal(3.5, 1.8, n_fraud)  # Mean ~33, lower than legit
        legit_amount = np.random.lognormal(4.2, 1.3, n_legit)  # Mean ~66, higher than fraud

        # Combine
        fraud_data = np.column_stack([fraud_features, fraud_time[:, np.newaxis], fraud_amount[:, np.newaxis]])
        legit_data = np.column_stack([legit_features, legit_time[:, np.newaxis], legit_amount[:, np.newaxis]])

        X = np.vstack([fraud_data, legit_data])
        y = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])

        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Create DataFrame
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        df = pd.DataFrame(X, columns=feature_names)

        # Apply preprocessing (creates additional features: Amount_Log, Amount_Bin, Hour, Time_Period)
        print("[INFO] Applying preprocessing to match training pipeline...")

        # The preprocessor is a dict with 'scaler' and potentially other info
        if isinstance(preprocessor, dict):
            if 'scaler' in preprocessor:
                scaler = preprocessor['scaler']
            else:
                print("[ERROR] Preprocessor format not recognized")
                return None, None, None
        else:
            scaler = preprocessor

        # Manually apply the same feature engineering as in training
        # (From src/data/creditcard_preprocessor.py)
        df_processed = df.copy()

        # Amount features
        df_processed['Amount_Log'] = np.log1p(df_processed['Amount'])
        df_processed['Amount_Bin'] = pd.cut(
            df_processed['Amount'],
            bins=[0, 50, 100, 300, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # Time features
        df_processed['Hour'] = (df_processed['Time'] / 3600) % 24
        df_processed['Time_Period'] = pd.cut(
            df_processed['Hour'],
            bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3],
            include_lowest=True
        ).astype(int)

        # Now we should have 34 features: V1-V28 (28) + Time + Amount + Amount_Log + Amount_Bin + Hour + Time_Period = 34
        print(f"[INFO] Features after preprocessing: {len(df_processed.columns)}")

        # Convert to array
        X_processed = df_processed.values

        # Apply scaling if scaler is available
        # The model was trained on scaled features, so we must scale the validation data too
        if scaler is not None:
            print(f"[INFO] Applying feature scaling...")
            X_processed = scaler.transform(X_processed)
        else:
            print(f"[WARNING] No scaler found - predictions may be inaccurate")

        # Predict risk scores
        risk_scores = model.predict_proba(X_processed)[:, 1]

        print(f"[SUCCESS] Loaded {len(y)} validation samples")
        print(f"  Fraud rate: {np.mean(y)*100:.2f}%")
        print(f"  Risk score range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")

        return risk_scores, y, df_processed

    def run_threshold_optimization(self, risk_scores: np.ndarray, y_true: np.ndarray):
        """
        Run threshold optimization.

        Args:
            risk_scores: Risk scores
            y_true: True labels
        """
        print("\n" + "="*70)
        print("TASK 1: THRESHOLD OPTIMIZATION")
        print("="*70)

        # Initialize tuner with business metrics
        tuner = ThresholdTuner(business_metrics=BusinessMetrics(
            fraud_loss_per_txn=500.0,      # Average fraud loss
            investigation_cost=5.0,         # Cost to investigate
            false_decline_cost=20.0,        # Cost of false decline
            fraud_base_rate=0.035           # Fraud rate (3.5% - matches validation data)
        ))

        # First, check what fraud catch rate is achievable
        print("\n[INFO] Analyzing achievable fraud catch rates...")

        # Get all results without constraints to see distribution
        all_results = tuner.grid_search_thresholds(y_true, risk_scores)
        max_fraud_catch = all_results['fraud_catch_rate'].max()

        print(f"[INFO] Maximum achievable fraud catch rate: {max_fraud_catch*100:.1f}%")
        print(f"[INFO] Risk score statistics:")
        print(f"      Fraud mean: {risk_scores[y_true==1].mean():.3f}")
        print(f"      Legit mean: {risk_scores[y_true==0].mean():.3f}")
        print(f"      Fraud median: {np.median(risk_scores[y_true==1]):.3f}")
        print(f"      Legit median: {np.median(risk_scores[y_true==0]):.3f}")

        # Adjust constraint based on what's achievable
        # If max achievable is less than 60%, use no constraint (just optimize for cost)
        if max_fraud_catch < 0.6:
            print(f"\n[WARNING] Maximum achievable fraud catch rate ({max_fraud_catch*100:.1f}%) is below 60%")
            print(f"          Optimizing for cost without fraud catch constraint")

            results = tuner.find_optimal_thresholds(
                y_true,
                risk_scores,
                constraints=None  # No constraints - just minimize cost
            )
        else:
            # Use 90% of maximum achievable, or 0.6 (60%), whichever is higher
            target_fraud_catch = max(0.6, max_fraud_catch * 0.9)

            print(f"\n[INFO] Using adjusted constraint: {target_fraud_catch*100:.1f}% fraud catch rate")
            print(f"      (90% of maximum achievable, minimum 60%)")

            # Find optimal thresholds with adjusted constraint
            results = tuner.find_optimal_thresholds(
                y_true,
                risk_scores,
                constraints={'min_fraud_catch_rate': target_fraud_catch}
            )

        # Generate recommendations
        recommendations = tuner.generate_recommendations(results)
        print(recommendations)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"threshold_optimization_{timestamp}.json"
        tuner.save_results(results, str(results_path))

        # Save plot
        plot_path = self.output_dir / f"threshold_optimization_{timestamp}.png"
        tuner.plot_threshold_surface(results['all_results'], save_path=str(plot_path))

        print(f"\n[SUCCESS] Threshold optimization complete!")
        print(f"  Results: {results_path}")
        print(f"  Plot: {plot_path}")

        return results

    def run_drift_detection(self, baseline_scores: np.ndarray, baseline_labels: np.ndarray):
        """
        Run model drift detection.

        Args:
            baseline_scores: Baseline (training) risk scores
            baseline_labels: Baseline true labels
        """
        print("\n" + "="*70)
        print("TASK 2: MODEL DRIFT DETECTION")
        print("="*70)

        # Initialize monitor
        monitor = ModelPerformanceMonitor(
            baseline_scores=baseline_scores,
            baseline_labels=baseline_labels,
            thresholds=DriftThresholds(
                psi_warning=0.1,
                psi_critical=0.25,
                ks_drop_warning=0.05,
                ks_drop_critical=0.10
            )
        )

        # Simulate production data with slight drift
        print("\n[INFO] Generating production data (with simulated drift)...")

        np.random.seed(44)
        n_prod = 3000
        fraud_rate = 0.002  # Slightly higher fraud rate (drift)

        n_fraud = int(n_prod * fraud_rate)
        n_legit = n_prod - n_fraud

        # Simulate drift: scores slightly shifted
        prod_fraud_scores = np.random.beta(7, 2.5, n_fraud)  # Slightly lower than baseline
        prod_legit_scores = np.random.beta(2.2, 8, n_legit)   # Slightly higher than baseline

        prod_labels = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
        prod_scores = np.concatenate([prod_fraud_scores, prod_legit_scores])

        # Shuffle
        shuffle_idx = np.random.permutation(n_prod)
        prod_labels = prod_labels[shuffle_idx]
        prod_scores = prod_scores[shuffle_idx]

        # Check drift
        drift_result = monitor.check_drift(prod_scores, prod_labels)

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"drift_report_{timestamp}.json"
        monitor.save_drift_report(drift_result, str(report_path))

        # Save plot
        plot_path = self.output_dir / f"drift_analysis_{timestamp}.png"
        monitor.plot_drift_analysis(prod_scores, prod_labels, save_path=str(plot_path))

        print(f"\n[SUCCESS] Drift detection complete!")
        print(f"  Report: {report_path}")
        print(f"  Plot: {plot_path}")

        return drift_result

    def publish_metrics_to_cloudwatch(self, threshold_results: dict, drift_results: dict):
        """
        Publish optimization results to CloudWatch.

        Args:
            threshold_results: Results from threshold optimization
            drift_results: Results from drift detection
        """
        print("\n" + "="*70)
        print("TASK 3: PUBLISH METRICS TO CLOUDWATCH")
        print("="*70)

        try:
            metrics = MetricsPublisher(namespace='VPBankFraud')

            # Publish threshold optimization metrics
            optimal = threshold_results['optimal_metrics']
            metrics.put_metric('OptimalPassThreshold', optimal['pass_threshold'])
            metrics.put_metric('OptimalBlockThreshold', optimal['block_threshold'])
            metrics.put_metric('OptimizationCostSavings', threshold_results['cost_savings'])

            # Publish drift detection metrics
            metrics.put_metric('PSI', drift_results['psi'])

            if drift_results['current_metrics']:
                metrics.put_metric('CurrentAUC', drift_results['current_metrics']['auc'])
                metrics.put_metric('CurrentKS', drift_results['current_metrics']['ks_statistic'])

            # Publish overall status (as numeric: 0=green, 1=yellow, 2=red)
            status_map = {'green': 0, 'yellow': 1, 'red': 2}
            metrics.put_metric('DriftStatus', status_map.get(drift_results['overall_status'], 0))

            # Flush all metrics
            metrics.flush()

            print("[SUCCESS] Metrics published to CloudWatch!")
            print("  Namespace: VPBankFraud")
            print("  Metrics: Thresholds, Costs, PSI, AUC, KS, Status")

        except Exception as e:
            print(f"[WARNING] Failed to publish to CloudWatch: {e}")
            print("  (This is expected if AWS credentials are not configured)")

    def run_all(self):
        """Run all optimization tasks."""

        # Load data
        risk_scores, y_true, features_df = self.load_validation_data()

        if risk_scores is None:
            print("\n[ERROR] Cannot proceed without validation data.")
            return

        # Task 1: Threshold Optimization
        threshold_results = self.run_threshold_optimization(risk_scores, y_true)

        # Task 2: Drift Detection
        drift_results = self.run_drift_detection(risk_scores, y_true)

        # Task 3: Publish to CloudWatch
        self.publish_metrics_to_cloudwatch(threshold_results, drift_results)

        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE!")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\n📋 Summary:")
        print(f"  • Optimal Pass Threshold: {threshold_results['optimal_thresholds']['pass_threshold']:.3f}")
        print(f"  • Optimal Block Threshold: {threshold_results['optimal_thresholds']['block_threshold']:.3f}")
        print(f"  • Cost Savings: ${threshold_results['cost_savings']:,.2f}")
        print(f"  • PSI: {drift_results['psi']:.4f} ({drift_results['psi_status']})")
        print(f"  • Drift Status: {drift_results['overall_status'].upper()}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run optimization tasks for VPBank fraud detection'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'threshold-only', 'drift-only'],
        help='Optimization mode (default: all)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Data directory (default: data/processed)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Models directory (default: models)'
    )

    args = parser.parse_args()

    # Run optimization
    orchestrator = OptimizationOrchestrator(
        data_dir=args.data_dir,
        models_dir=args.models_dir
    )

    if args.mode == 'all':
        orchestrator.run_all()

    elif args.mode == 'threshold-only':
        risk_scores, y_true, _ = orchestrator.load_validation_data()
        if risk_scores is not None:
            orchestrator.run_threshold_optimization(risk_scores, y_true)

    elif args.mode == 'drift-only':
        risk_scores, y_true, _ = orchestrator.load_validation_data()
        if risk_scores is not None:
            orchestrator.run_drift_detection(risk_scores, y_true)


if __name__ == "__main__":
    main()
