"""
Threshold Tuning Module for VPBank StreamGuard

Optimizes RBA policy thresholds based on business metrics:
- Fraud loss cost
- Investigation cost (challenge)
- False decline cost (blocking legitimate transactions)

Supports:
- Cost-based threshold optimization
- Channel-specific tuning
- Time-of-day specific tuning
- Precision-Recall curve analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BusinessMetrics:
    """Business cost metrics for threshold optimization."""
    fraud_loss_per_txn: float = 500.0  # Average loss per fraud transaction
    investigation_cost: float = 5.0     # Cost to investigate challenged transaction
    false_decline_cost: float = 20.0    # Cost of blocking legitimate transaction (customer churn)
    fraud_base_rate: float = 0.035      # Base fraud rate (3.5% from IEEE-CIS)


@dataclass
class ThresholdConfig:
    """Threshold configuration for RBA policy."""
    pass_threshold: float = 0.3    # Below this: pass
    block_threshold: float = 0.7   # Above this: block
    channel_adjustments: Dict[str, float] = None
    time_adjustments: Dict[str, float] = None

    def __post_init__(self):
        if self.channel_adjustments is None:
            self.channel_adjustments = {
                'online_banking': 0.0,
                'mobile_app': -0.05,
                'atm': 0.05,
                'pos': 0.03,
                'card_not_present': 0.10
            }
        if self.time_adjustments is None:
            self.time_adjustments = {
                '0-6': 0.10,    # Late night
                '6-9': 0.05,    # Early morning
                '9-18': 0.00,   # Business hours
                '18-22': 0.03,  # Evening
                '22-24': 0.08   # Late night
            }


class ThresholdTuner:
    """
    Threshold tuning and optimization for fraud detection system.

    Features:
    - Cost-based optimization
    - Precision-Recall analysis
    - Channel/time-specific recommendations
    - ROI calculation
    """

    def __init__(self, business_metrics: Optional[BusinessMetrics] = None):
        """
        Initialize threshold tuner.

        Args:
            business_metrics: Business cost metrics for optimization
        """
        self.business_metrics = business_metrics or BusinessMetrics()
        self.results = {}

    def calculate_expected_cost(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        pass_threshold: float,
        block_threshold: float
    ) -> Dict[str, float]:
        """
        Calculate expected cost for given thresholds.

        Args:
            y_true: True labels (1=fraud, 0=legitimate)
            risk_scores: Risk scores (0-1)
            pass_threshold: Threshold below which to pass
            block_threshold: Threshold above which to block

        Returns:
            Dict with cost breakdown
        """
        # Classify transactions
        pass_mask = risk_scores < pass_threshold
        challenge_mask = (risk_scores >= pass_threshold) & (risk_scores < block_threshold)
        block_mask = risk_scores >= block_threshold

        # Calculate outcomes
        # Pass zone: undetected fraud
        passed_fraud = np.sum((pass_mask) & (y_true == 1))
        passed_legit = np.sum((pass_mask) & (y_true == 0))

        # Challenge zone: investigated (catch some fraud, cost to investigate)
        challenged_fraud = np.sum((challenge_mask) & (y_true == 1))
        challenged_legit = np.sum((challenge_mask) & (y_true == 0))

        # Block zone: blocked fraud (good), blocked legit (false decline cost)
        blocked_fraud = np.sum((block_mask) & (y_true == 1))
        blocked_legit = np.sum((block_mask) & (y_true == 0))

        # Calculate costs
        fraud_loss_cost = passed_fraud * self.business_metrics.fraud_loss_per_txn
        investigation_cost = (challenged_fraud + challenged_legit) * self.business_metrics.investigation_cost
        false_decline_cost = blocked_legit * self.business_metrics.false_decline_cost

        total_cost = fraud_loss_cost + investigation_cost + false_decline_cost

        # Calculate rates
        total_txns = len(y_true)
        total_fraud = np.sum(y_true == 1)

        fraud_catch_rate = (challenged_fraud + blocked_fraud) / total_fraud if total_fraud > 0 else 0
        false_positive_rate = (challenged_legit + blocked_legit) / (total_txns - total_fraud)

        return {
            'total_cost': total_cost,
            'fraud_loss_cost': fraud_loss_cost,
            'investigation_cost': investigation_cost,
            'false_decline_cost': false_decline_cost,
            'passed_fraud': int(passed_fraud),
            'passed_legit': int(passed_legit),
            'challenged_fraud': int(challenged_fraud),
            'challenged_legit': int(challenged_legit),
            'blocked_fraud': int(blocked_fraud),
            'blocked_legit': int(blocked_legit),
            'fraud_catch_rate': fraud_catch_rate,
            'false_positive_rate': false_positive_rate,
            'pass_threshold': pass_threshold,
            'block_threshold': block_threshold
        }

    def grid_search_thresholds(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        pass_range: Tuple[float, float] = (0.1, 0.5),
        block_range: Tuple[float, float] = (0.5, 0.9),
        n_points: int = 20
    ) -> pd.DataFrame:
        """
        Grid search over threshold combinations to find optimal thresholds.

        Args:
            y_true: True labels
            risk_scores: Risk scores
            pass_range: (min, max) for pass threshold
            block_range: (min, max) for block threshold
            n_points: Number of points to try for each threshold

        Returns:
            DataFrame with all combinations and their costs
        """
        print(f"\n[INFO] Starting grid search over {n_points}x{n_points} threshold combinations...")

        pass_thresholds = np.linspace(pass_range[0], pass_range[1], n_points)
        block_thresholds = np.linspace(block_range[0], block_range[1], n_points)

        results = []

        for pass_thresh in pass_thresholds:
            for block_thresh in block_thresholds:
                if pass_thresh >= block_thresh:
                    continue  # Invalid: pass threshold must be < block threshold

                cost_result = self.calculate_expected_cost(
                    y_true, risk_scores, pass_thresh, block_thresh
                )
                results.append(cost_result)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('total_cost')

        print(f"[SUCCESS] Grid search complete! Found {len(df_results)} valid combinations.")

        return df_results

    def find_optimal_thresholds(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        constraints: Optional[Dict[str, float]] = None
    ) -> Dict[str, any]:
        """
        Find optimal thresholds that minimize total cost.

        Args:
            y_true: True labels
            risk_scores: Risk scores
            constraints: Optional constraints (e.g., min_fraud_catch_rate=0.80)

        Returns:
            Dict with optimal thresholds and analysis
        """
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION")
        print("="*60)

        # Grid search
        df_results = self.grid_search_thresholds(y_true, risk_scores)

        # Apply constraints if provided
        if constraints:
            print(f"\n[INFO] Applying constraints: {constraints}")
            for key, value in constraints.items():
                if key == 'min_fraud_catch_rate':
                    df_results = df_results[df_results['fraud_catch_rate'] >= value]
                elif key == 'max_false_positive_rate':
                    df_results = df_results[df_results['false_positive_rate'] <= value]

            print(f"[INFO] {len(df_results)} combinations meet constraints")

        if len(df_results) == 0:
            raise ValueError("No threshold combinations meet the specified constraints")

        # Get optimal
        optimal = df_results.iloc[0].to_dict()

        # Get current baseline (0.3, 0.7)
        baseline = self.calculate_expected_cost(y_true, risk_scores, 0.3, 0.7)

        # Calculate improvement
        cost_savings = baseline['total_cost'] - optimal['total_cost']
        cost_savings_pct = (cost_savings / baseline['total_cost']) * 100

        print("\n" + "-"*60)
        print("OPTIMIZATION RESULTS")
        print("-"*60)
        print(f"\n📊 BASELINE THRESHOLDS (pass=0.3, block=0.7):")
        print(f"  Total Cost: ${baseline['total_cost']:,.2f}")
        print(f"  - Fraud Loss: ${baseline['fraud_loss_cost']:,.2f}")
        print(f"  - Investigation: ${baseline['investigation_cost']:,.2f}")
        print(f"  - False Declines: ${baseline['false_decline_cost']:,.2f}")
        print(f"  Fraud Catch Rate: {baseline['fraud_catch_rate']*100:.2f}%")
        print(f"  False Positive Rate: {baseline['false_positive_rate']*100:.2f}%")

        print(f"\n✅ OPTIMAL THRESHOLDS (pass={optimal['pass_threshold']:.3f}, block={optimal['block_threshold']:.3f}):")
        print(f"  Total Cost: ${optimal['total_cost']:,.2f}")
        print(f"  - Fraud Loss: ${optimal['fraud_loss_cost']:,.2f}")
        print(f"  - Investigation: ${optimal['investigation_cost']:,.2f}")
        print(f"  - False Declines: ${optimal['false_decline_cost']:,.2f}")
        print(f"  Fraud Catch Rate: {optimal['fraud_catch_rate']*100:.2f}%")
        print(f"  False Positive Rate: {optimal['false_positive_rate']*100:.2f}%")

        print(f"\n💰 COST SAVINGS: ${cost_savings:,.2f} ({cost_savings_pct:+.1f}%)")
        print("="*60 + "\n")

        return {
            'optimal_thresholds': {
                'pass_threshold': optimal['pass_threshold'],
                'block_threshold': optimal['block_threshold']
            },
            'optimal_metrics': optimal,
            'baseline_metrics': baseline,
            'cost_savings': cost_savings,
            'cost_savings_pct': cost_savings_pct,
            'all_results': df_results
        }

    def analyze_by_segment(
        self,
        y_true: np.ndarray,
        risk_scores: np.ndarray,
        segment_col: np.ndarray,
        segment_name: str = "channel"
    ) -> Dict[str, Dict]:
        """
        Analyze optimal thresholds by segment (e.g., channel, time-of-day).

        Args:
            y_true: True labels
            risk_scores: Risk scores
            segment_col: Segment identifiers
            segment_name: Name of segment (for display)

        Returns:
            Dict with optimal thresholds per segment
        """
        print(f"\n[INFO] Analyzing thresholds by {segment_name}...")

        segments = np.unique(segment_col)
        results = {}

        for segment in segments:
            mask = segment_col == segment

            if np.sum(mask) < 100:  # Skip segments with too few samples
                print(f"[SKIP] {segment_name}={segment}: Only {np.sum(mask)} samples")
                continue

            print(f"\n--- {segment_name.upper()}: {segment} ({np.sum(mask)} samples) ---")

            optimal = self.find_optimal_thresholds(
                y_true[mask],
                risk_scores[mask]
            )

            results[str(segment)] = optimal

        return results

    def plot_threshold_surface(
        self,
        df_results: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot cost surface over threshold combinations.

        Args:
            df_results: DataFrame from grid_search_thresholds()
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Pivot for heatmap
        pivot_cost = df_results.pivot(
            index='block_threshold',
            columns='pass_threshold',
            values='total_cost'
        )

        pivot_fraud_catch = df_results.pivot(
            index='block_threshold',
            columns='pass_threshold',
            values='fraud_catch_rate'
        )

        pivot_fp = df_results.pivot(
            index='block_threshold',
            columns='pass_threshold',
            values='false_positive_rate'
        )

        # Total Cost
        sns.heatmap(pivot_cost, ax=axes[0, 0], cmap='RdYlGn_r', cbar_kws={'label': 'Total Cost ($)'})
        axes[0, 0].set_title('Total Cost by Threshold Combination')
        axes[0, 0].set_xlabel('Pass Threshold')
        axes[0, 0].set_ylabel('Block Threshold')

        # Fraud Catch Rate
        sns.heatmap(pivot_fraud_catch, ax=axes[0, 1], cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'Fraud Catch Rate'})
        axes[0, 1].set_title('Fraud Catch Rate by Threshold Combination')
        axes[0, 1].set_xlabel('Pass Threshold')
        axes[0, 1].set_ylabel('Block Threshold')

        # False Positive Rate
        sns.heatmap(pivot_fp, ax=axes[1, 0], cmap='RdYlGn_r', vmin=0, vmax=0.1,
                    cbar_kws={'label': 'False Positive Rate'})
        axes[1, 0].set_title('False Positive Rate by Threshold Combination')
        axes[1, 0].set_xlabel('Pass Threshold')
        axes[1, 0].set_ylabel('Block Threshold')

        # Cost breakdown
        optimal = df_results.iloc[0]
        baseline = df_results[(df_results['pass_threshold'].round(2) == 0.30) &
                              (df_results['block_threshold'].round(2) == 0.70)]

        if len(baseline) > 0:
            baseline = baseline.iloc[0]

            categories = ['Fraud Loss', 'Investigation', 'False Decline', 'TOTAL']
            baseline_costs = [
                baseline['fraud_loss_cost'],
                baseline['investigation_cost'],
                baseline['false_decline_cost'],
                baseline['total_cost']
            ]
            optimal_costs = [
                optimal['fraud_loss_cost'],
                optimal['investigation_cost'],
                optimal['false_decline_cost'],
                optimal['total_cost']
            ]

            x = np.arange(len(categories))
            width = 0.35

            axes[1, 1].bar(x - width/2, baseline_costs, width, label='Baseline (0.3, 0.7)', alpha=0.8)
            axes[1, 1].bar(x + width/2, optimal_costs, width, label=f'Optimal ({optimal["pass_threshold"]:.2f}, {optimal["block_threshold"]:.2f})', alpha=0.8)

            axes[1, 1].set_ylabel('Cost ($)')
            axes[1, 1].set_title('Cost Comparison: Baseline vs Optimal')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(categories, rotation=15)
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SUCCESS] Plot saved to {save_path}")

        plt.show()

    def generate_recommendations(
        self,
        optimal_results: Dict,
        current_config: Optional[ThresholdConfig] = None
    ) -> str:
        """
        Generate human-readable recommendations for threshold updates.

        Args:
            optimal_results: Results from find_optimal_thresholds()
            current_config: Current threshold configuration

        Returns:
            Formatted recommendation text
        """
        current_config = current_config or ThresholdConfig()

        optimal = optimal_results['optimal_thresholds']
        metrics = optimal_results['optimal_metrics']
        baseline = optimal_results['baseline_metrics']
        savings = optimal_results['cost_savings']
        savings_pct = optimal_results['cost_savings_pct']

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          THRESHOLD OPTIMIZATION RECOMMENDATIONS              ║
╚══════════════════════════════════════════════════════════════╝

📋 CURRENT CONFIGURATION
─────────────────────────────────────────────────────────────
  Pass Threshold:    {current_config.pass_threshold:.3f}
  Block Threshold:   {current_config.block_threshold:.3f}

  Current Performance:
  • Total Cost:           ${baseline['total_cost']:,.2f}
  • Fraud Catch Rate:     {baseline['fraud_catch_rate']*100:.1f}%
  • False Positive Rate:  {baseline['false_positive_rate']*100:.2f}%

✅ RECOMMENDED CONFIGURATION
─────────────────────────────────────────────────────────────
  Pass Threshold:    {optimal['pass_threshold']:.3f}  (change: {optimal['pass_threshold'] - current_config.pass_threshold:+.3f})
  Block Threshold:   {optimal['block_threshold']:.3f}  (change: {optimal['block_threshold'] - current_config.block_threshold:+.3f})

  Expected Performance:
  • Total Cost:           ${metrics['total_cost']:,.2f}  ({savings_pct:+.1f}%)
  • Fraud Catch Rate:     {metrics['fraud_catch_rate']*100:.1f}%  ({(metrics['fraud_catch_rate']-baseline['fraud_catch_rate'])*100:+.1f}pp)
  • False Positive Rate:  {metrics['false_positive_rate']*100:.2f}%  ({(metrics['false_positive_rate']-baseline['false_positive_rate'])*100:+.2f}pp)

💰 EXPECTED IMPACT
─────────────────────────────────────────────────────────────
  Annual Cost Savings:   ${savings * 12:,.2f}  (estimated, assuming monthly analysis)

  Cost Breakdown Changes:
  • Fraud Loss:      ${baseline['fraud_loss_cost']:>10,.0f} → ${metrics['fraud_loss_cost']:>10,.0f}  ({(metrics['fraud_loss_cost']-baseline['fraud_loss_cost'])/baseline['fraud_loss_cost']*100:+.1f}%)
  • Investigation:   ${baseline['investigation_cost']:>10,.0f} → ${metrics['investigation_cost']:>10,.0f}  ({(metrics['investigation_cost']-baseline['investigation_cost'])/baseline['investigation_cost']*100:+.1f}%)
  • False Declines:  ${baseline['false_decline_cost']:>10,.0f} → ${metrics['false_decline_cost']:>10,.0f}  ({(metrics['false_decline_cost']-baseline['false_decline_cost'])/baseline['false_decline_cost']*100:+.1f}%)

🔧 IMPLEMENTATION STEPS
─────────────────────────────────────────────────────────────
1. Update RBA policy configuration:
   - Set pass_threshold = {optimal['pass_threshold']:.3f}
   - Set block_threshold = {optimal['block_threshold']:.3f}

2. Deploy to staging environment for testing

3. Monitor for 1-2 weeks:
   - Track actual fraud catch rate
   - Monitor false positive complaints
   - Measure investigation workload

4. If successful, promote to production

5. Re-run optimization monthly to adapt to changing patterns

⚠️  CONSIDERATIONS
─────────────────────────────────────────────────────────────
• Results based on historical data - actual performance may vary
• Consider A/B testing before full rollout
• Monitor customer feedback closely after deployment
• Adjust channel/time-specific thresholds separately if needed

"""
        return report

    def save_results(self, results: Dict, output_path: str):
        """Save optimization results to JSON."""
        # Convert numpy/pandas types to native Python types
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

        # Create serializable version
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: convert_types(v) for k, v in value.items()}
            else:
                serializable_results[key] = convert_types(value)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"[SUCCESS] Results saved to {output_path}")


if __name__ == "__main__":
    """Demo: Threshold optimization on synthetic data."""

    print("\n" + "="*60)
    print("THRESHOLD TUNER - DEMO")
    print("="*60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    fraud_rate = 0.035

    # Simulate risk scores
    # Fraud transactions: higher risk scores
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    fraud_scores = np.random.beta(8, 2, n_fraud)  # Skewed towards high scores
    legit_scores = np.random.beta(2, 8, n_legit)  # Skewed towards low scores

    y_true = np.concatenate([np.ones(n_fraud), np.zeros(n_legit)])
    risk_scores = np.concatenate([fraud_scores, legit_scores])

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    y_true = y_true[shuffle_idx]
    risk_scores = risk_scores[shuffle_idx]

    # Run optimization
    tuner = ThresholdTuner(business_metrics=BusinessMetrics(
        fraud_loss_per_txn=500,
        investigation_cost=5,
        false_decline_cost=20
    ))

    results = tuner.find_optimal_thresholds(
        y_true,
        risk_scores,
        constraints={'min_fraud_catch_rate': 0.80}
    )

    # Generate recommendations
    recommendations = tuner.generate_recommendations(results)
    print(recommendations)

    # Plot results
    tuner.plot_threshold_surface(results['all_results'])

    print("\n[SUCCESS] Demo complete!")
