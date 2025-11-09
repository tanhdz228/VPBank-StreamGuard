"""
Model Performance Monitoring for VPBank StreamGuard

Tracks model performance over time using:
- PSI (Population Stability Index): Detects distribution drift
- KS Statistic: Measures discrimination power
- Performance metrics: AUC, Precision, Recall
- Drift detection and alerting

Usage:
    from src.monitoring.model_performance import ModelPerformanceMonitor

    monitor = ModelPerformanceMonitor(baseline_scores, baseline_labels)
    drift_report = monitor.check_drift(production_scores, production_labels)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    psi_warning: float = 0.1   # Moderate drift
    psi_critical: float = 0.25  # Significant drift - retrain recommended
    ks_drop_warning: float = 0.05  # 5% drop in KS
    ks_drop_critical: float = 0.10  # 10% drop in KS
    auc_drop_warning: float = 0.02   # 2% drop in AUC
    auc_drop_critical: float = 0.05  # 5% drop in AUC


class ModelPerformanceMonitor:
    """
    Monitor model performance and detect drift over time.

    Features:
    - PSI calculation for score distribution drift
    - KS statistic for discrimination power
    - Performance metrics tracking
    - Drift alerts and recommendations
    """

    def __init__(
        self,
        baseline_scores: Optional[np.ndarray] = None,
        baseline_labels: Optional[np.ndarray] = None,
        thresholds: Optional[DriftThresholds] = None
    ):
        """
        Initialize performance monitor.

        Args:
            baseline_scores: Baseline (training/validation) risk scores
            baseline_labels: Baseline true labels
            thresholds: Drift detection thresholds
        """
        self.baseline_scores = baseline_scores
        self.baseline_labels = baseline_labels
        self.thresholds = thresholds or DriftThresholds()

        # Calculate baseline metrics if data provided
        self.baseline_metrics = None
        if baseline_scores is not None and baseline_labels is not None:
            self.baseline_metrics = self._calculate_metrics(baseline_scores, baseline_labels)

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in distribution between two datasets.
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.25: Moderate change (monitor closely)
        - PSI ≥ 0.25: Significant change (retrain model)

        Args:
            expected: Baseline distribution (e.g., training scores)
            actual: Current distribution (e.g., production scores)
            bins: Number of bins for discretization

        Returns:
            Tuple of (PSI value, breakdown DataFrame)
        """
        # Create bins
        breakpoints = np.linspace(0, 1, bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Bin the data
        expected_bins = np.histogram(expected, bins=breakpoints)[0]
        actual_bins = np.histogram(actual, bins=breakpoints)[0]

        # Calculate proportions (add small epsilon to avoid log(0))
        epsilon = 1e-10
        expected_props = (expected_bins + epsilon) / (len(expected) + epsilon * bins)
        actual_props = (actual_bins + epsilon) / (len(actual) + epsilon * bins)

        # Calculate PSI
        psi_values = (actual_props - expected_props) * np.log(actual_props / expected_props)
        psi = np.sum(psi_values)

        # Create breakdown dataframe
        breakdown = pd.DataFrame({
            'bin': range(bins),
            'expected_count': expected_bins,
            'actual_count': actual_bins,
            'expected_prop': expected_props,
            'actual_prop': actual_props,
            'psi_contribution': psi_values
        })

        return psi, breakdown

    def calculate_ks_statistic(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate Kolmogorov-Smirnov (KS) statistic.

        KS measures the maximum separation between cumulative distributions
        of fraud and legitimate scores. Higher KS = better discrimination.

        Args:
            y_true: True labels (1=fraud, 0=legitimate)
            y_scores: Risk scores (0-1)

        Returns:
            Dict with KS statistic and threshold
        """
        # Separate scores by class
        fraud_scores = y_scores[y_true == 1]
        legit_scores = y_scores[y_true == 0]

        # Sort scores
        thresholds = np.linspace(0, 1, 1000)

        # Calculate cumulative distributions
        fraud_cdf = np.array([np.mean(fraud_scores <= t) for t in thresholds])
        legit_cdf = np.array([np.mean(legit_scores <= t) for t in thresholds])

        # KS statistic is max separation
        ks_values = fraud_cdf - legit_cdf
        ks_stat = np.max(ks_values)
        ks_threshold = thresholds[np.argmax(ks_values)]

        return {
            'ks_statistic': ks_stat,
            'ks_threshold': ks_threshold,
            'fraud_cdf': fraud_cdf,
            'legit_cdf': legit_cdf,
            'thresholds': thresholds
        }

    def _calculate_metrics(
        self,
        y_scores: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""

        # AUC
        auc = roc_auc_score(y_true, y_scores)

        # KS statistic
        ks_result = self.calculate_ks_statistic(y_true, y_scores)

        # Precision-Recall at different thresholds
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

        # Find precision/recall at typical thresholds
        thresholds_to_check = [0.3, 0.5, 0.7]
        threshold_metrics = {}

        for thresh in thresholds_to_check:
            y_pred = (y_scores >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            threshold_metrics[f'precision@{thresh}'] = prec
            threshold_metrics[f'recall@{thresh}'] = rec

        return {
            'auc': auc,
            'ks_statistic': ks_result['ks_statistic'],
            'ks_threshold': ks_result['ks_threshold'],
            **threshold_metrics,
            'n_samples': len(y_true),
            'fraud_rate': np.mean(y_true)
        }

    def check_drift(
        self,
        current_scores: np.ndarray,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, any]:
        """
        Check for distribution drift and performance degradation.

        Args:
            current_scores: Current production risk scores
            current_labels: Current true labels (if available)

        Returns:
            Dict with drift analysis and recommendations
        """
        if self.baseline_scores is None:
            raise ValueError("Baseline scores not set. Initialize with baseline data.")

        print("\n" + "="*60)
        print("MODEL DRIFT DETECTION")
        print("="*60)

        # Calculate PSI
        psi, psi_breakdown = self.calculate_psi(self.baseline_scores, current_scores)

        print(f"\n📊 POPULATION STABILITY INDEX (PSI)")
        print(f"   PSI: {psi:.4f}")

        if psi < self.thresholds.psi_warning:
            psi_status = "✅ STABLE"
            psi_severity = "green"
        elif psi < self.thresholds.psi_critical:
            psi_status = "⚠️  MODERATE DRIFT"
            psi_severity = "yellow"
        else:
            psi_status = "🚨 SIGNIFICANT DRIFT"
            psi_severity = "red"

        print(f"   Status: {psi_status}")

        # Performance metrics (if labels available)
        current_metrics = None
        performance_drift = {}

        if current_labels is not None and self.baseline_labels is not None:
            current_metrics = self._calculate_metrics(current_scores, current_labels)

            print(f"\n📈 PERFORMANCE COMPARISON")
            print(f"   {'Metric':<20} {'Baseline':<12} {'Current':<12} {'Change':<12}")
            print(f"   {'-'*56}")

            for metric in ['auc', 'ks_statistic']:
                baseline_val = self.baseline_metrics[metric]
                current_val = current_metrics[metric]
                change = current_val - baseline_val
                change_pct = (change / baseline_val) * 100 if baseline_val != 0 else 0

                print(f"   {metric:<20} {baseline_val:<12.4f} {current_val:<12.4f} {change:+.4f} ({change_pct:+.1f}%)")

                performance_drift[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'change': change,
                    'change_pct': change_pct
                }

        # Determine overall status and recommendations
        recommendations = []
        overall_status = "green"

        if psi_severity == "red":
            overall_status = "red"
            recommendations.append("🔴 URGENT: Significant distribution drift detected. Model retraining recommended.")

        elif psi_severity == "yellow":
            overall_status = "yellow"
            recommendations.append("🟡 WARNING: Moderate drift detected. Monitor closely and consider retraining if drift persists.")

        if current_metrics:
            auc_change = performance_drift['auc']['change']
            ks_change = performance_drift['ks_statistic']['change']

            if auc_change < -self.thresholds.auc_drop_critical or ks_change < -self.thresholds.ks_drop_critical:
                overall_status = "red"
                recommendations.append("🔴 URGENT: Significant performance degradation. Immediate retraining required.")

            elif auc_change < -self.thresholds.auc_drop_warning or ks_change < -self.thresholds.ks_drop_warning:
                if overall_status == "green":
                    overall_status = "yellow"
                recommendations.append("🟡 WARNING: Performance degradation detected. Schedule retraining soon.")

        if overall_status == "green":
            recommendations.append("✅ Model performance is stable. Continue monitoring.")

        print(f"\n🎯 OVERALL STATUS: {overall_status.upper()}")
        print(f"\n📝 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")

        print("="*60 + "\n")

        return {
            'psi': psi,
            'psi_breakdown': psi_breakdown,
            'psi_status': psi_severity,
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'performance_drift': performance_drift,
            'overall_status': overall_status,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }

    def plot_drift_analysis(
        self,
        current_scores: np.ndarray,
        current_labels: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Create visualizations for drift analysis.

        Args:
            current_scores: Current production scores
            current_labels: Current labels (optional)
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Score distribution comparison
        axes[0, 0].hist(self.baseline_scores, bins=50, alpha=0.5, label='Baseline', density=True)
        axes[0, 0].hist(current_scores, bins=50, alpha=0.5, label='Current', density=True)
        axes[0, 0].set_xlabel('Risk Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Score Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # 2. PSI breakdown
        psi, psi_breakdown = self.calculate_psi(self.baseline_scores, current_scores)

        axes[0, 1].bar(psi_breakdown['bin'], psi_breakdown['psi_contribution'], alpha=0.7)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[0, 1].set_xlabel('Score Bin')
        axes[0, 1].set_ylabel('PSI Contribution')
        axes[0, 1].set_title(f'PSI Breakdown (Total: {psi:.4f})')
        axes[0, 1].grid(alpha=0.3)

        # 3. CDF comparison
        baseline_sorted = np.sort(self.baseline_scores)
        current_sorted = np.sort(current_scores)

        baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
        current_cdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)

        axes[1, 0].plot(baseline_sorted, baseline_cdf, label='Baseline', alpha=0.7)
        axes[1, 0].plot(current_sorted, current_cdf, label='Current', alpha=0.7)
        axes[1, 0].set_xlabel('Risk Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Performance metrics (if labels available)
        if current_labels is not None and self.baseline_labels is not None:
            baseline_metrics = self.baseline_metrics
            current_metrics = self._calculate_metrics(current_scores, current_labels)

            metrics = ['auc', 'ks_statistic']
            baseline_vals = [baseline_metrics[m] for m in metrics]
            current_vals = [current_metrics[m] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            axes[1, 1].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
            axes[1, 1].bar(x + width/2, current_vals, width, label='Current', alpha=0.8)

            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('Performance Metrics Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3, axis='y')

        else:
            axes[1, 1].text(0.5, 0.5, 'Labels not available\nfor performance comparison',
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] Drift analysis plot saved to {save_path}")

        plt.show()

    def save_drift_report(self, drift_result: Dict, output_path: str):
        """Save drift detection results to JSON."""

        # Convert non-serializable objects
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            return obj

        serializable_result = {}
        for key, value in drift_result.items():
            if isinstance(value, dict):
                serializable_result[key] = {k: convert_types(v) for k, v in value.items()}
            else:
                serializable_result[key] = convert_types(value)

        with open(output_path, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)

        print(f"[SUCCESS] Drift report saved to {output_path}")


if __name__ == "__main__":
    """Demo: Model performance monitoring and drift detection."""

    print("\n" + "="*60)
    print("MODEL PERFORMANCE MONITORING - DEMO")
    print("="*60)

    # Generate synthetic baseline data
    np.random.seed(42)
    n_baseline = 10000
    fraud_rate = 0.035

    # Baseline: fraud and legitimate scores
    n_fraud_baseline = int(n_baseline * fraud_rate)
    n_legit_baseline = n_baseline - n_fraud_baseline

    baseline_fraud_scores = np.random.beta(8, 2, n_fraud_baseline)
    baseline_legit_scores = np.random.beta(2, 8, n_legit_baseline)

    baseline_labels = np.concatenate([np.ones(n_fraud_baseline), np.zeros(n_legit_baseline)])
    baseline_scores = np.concatenate([baseline_fraud_scores, baseline_legit_scores])

    # Shuffle
    shuffle_idx = np.random.permutation(n_baseline)
    baseline_labels = baseline_labels[shuffle_idx]
    baseline_scores = baseline_scores[shuffle_idx]

    # Current data with drift (shift distribution)
    n_current = 5000
    n_fraud_current = int(n_current * fraud_rate)
    n_legit_current = n_current - n_fraud_current

    # Simulate drift: fraud scores slightly lower (model degradation)
    current_fraud_scores = np.random.beta(6, 3, n_fraud_current)  # Lower discrimination
    current_legit_scores = np.random.beta(2.5, 7.5, n_legit_current)  # Slight shift

    current_labels = np.concatenate([np.ones(n_fraud_current), np.zeros(n_legit_current)])
    current_scores = np.concatenate([current_fraud_scores, current_legit_scores])

    shuffle_idx = np.random.permutation(n_current)
    current_labels = current_labels[shuffle_idx]
    current_scores = current_scores[shuffle_idx]

    # Initialize monitor
    monitor = ModelPerformanceMonitor(
        baseline_scores=baseline_scores,
        baseline_labels=baseline_labels
    )

    # Check drift
    drift_result = monitor.check_drift(current_scores, current_labels)

    # Plot analysis
    monitor.plot_drift_analysis(current_scores, current_labels)

    print("\n[SUCCESS] Demo complete!")
