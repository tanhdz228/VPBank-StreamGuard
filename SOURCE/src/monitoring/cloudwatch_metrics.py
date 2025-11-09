"""
CloudWatch Metrics Module for VPBank StreamGuard

Provides custom metrics publishing to CloudWatch for:
- Model performance (risk score distribution)
- Decision outcomes (pass/challenge/block rates)
- Entity risk statistics
- Latency tracking
- Error rates

Usage in Lambda:
    from src.monitoring.cloudwatch_metrics import MetricsPublisher

    metrics = MetricsPublisher(namespace='VPBankFraud')
    metrics.put_metric('RiskScore', risk_score, unit='None')
    metrics.put_metric('Latency', latency_ms, unit='Milliseconds')
"""

import boto3
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from collections import defaultdict


class MetricsPublisher:
    """
    CloudWatch metrics publisher for fraud detection system.

    Features:
    - Custom metrics with dimensions
    - Batch publishing (up to 20 metrics per API call)
    - Automatic metric aggregation
    - Cost-optimized publishing
    """

    def __init__(self, namespace: str = 'VPBankFraud', region: str = 'us-east-1'):
        """
        Initialize metrics publisher.

        Args:
            namespace: CloudWatch namespace for custom metrics
            region: AWS region
        """
        self.namespace = namespace
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.metrics_buffer = []
        self.max_buffer_size = 20  # CloudWatch limit per PutMetricData call

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = 'None',
        dimensions: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Add a metric to the buffer.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: CloudWatch unit (None, Count, Milliseconds, etc.)
            dimensions: Optional dimensions (e.g., {'Channel': 'mobile_app'})
            timestamp: Optional timestamp (defaults to now)
        """
        metric_data = {
            'MetricName': metric_name,
            'Value': value,
            'Unit': unit,
            'Timestamp': timestamp or datetime.utcnow()
        }

        if dimensions:
            metric_data['Dimensions'] = [
                {'Name': k, 'Value': str(v)} for k, v in dimensions.items()
            ]

        self.metrics_buffer.append(metric_data)

        # Auto-flush if buffer is full
        if len(self.metrics_buffer) >= self.max_buffer_size:
            self.flush()

    def flush(self):
        """Publish all buffered metrics to CloudWatch."""
        if not self.metrics_buffer:
            return

        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=self.metrics_buffer
            )
            print(f"[METRICS] Published {len(self.metrics_buffer)} metrics to CloudWatch")
            self.metrics_buffer = []
        except Exception as e:
            print(f"[ERROR] Failed to publish metrics: {e}")
            # Clear buffer to prevent memory buildup
            self.metrics_buffer = []

    def record_prediction(
        self,
        risk_score: float,
        model_score: float,
        entity_risk: float,
        decision: str,
        latency_ms: float,
        channel: Optional[str] = None
    ):
        """
        Record a complete prediction event with all relevant metrics.

        Args:
            risk_score: Combined risk score
            model_score: Model-only score
            entity_risk: Entity risk score
            decision: Decision (pass/challenge/block)
            latency_ms: Processing latency in milliseconds
            channel: Optional channel (mobile_app, online_banking, etc.)
        """
        dimensions = {'Channel': channel} if channel else {}

        # Core metrics
        self.put_metric('RiskScore', risk_score, unit='None', dimensions=dimensions)
        self.put_metric('ModelScore', model_score, unit='None', dimensions=dimensions)
        self.put_metric('EntityRisk', entity_risk, unit='None', dimensions=dimensions)
        self.put_metric('Latency', latency_ms, unit='Milliseconds', dimensions=dimensions)

        # Decision metrics (as counters)
        self.put_metric('PredictionCount', 1, unit='Count', dimensions=dimensions)

        decision_dims = {**dimensions, 'Decision': decision}
        self.put_metric('DecisionCount', 1, unit='Count', dimensions=decision_dims)

        # Auto-flush
        self.flush()

    def record_error(self, error_type: str, error_message: str):
        """
        Record an error event.

        Args:
            error_type: Type of error (ModelLoadError, DynamoDBError, etc.)
            error_message: Error message
        """
        dimensions = {'ErrorType': error_type}
        self.put_metric('ErrorCount', 1, unit='Count', dimensions=dimensions)
        self.flush()

        print(f"[ERROR] {error_type}: {error_message}")


class PerformanceMonitor:
    """
    Monitor model performance metrics over time.

    Tracks:
    - Risk score distribution (mean, p50, p95, p99)
    - Decision rates
    - Latency percentiles
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.risk_scores = []
        self.latencies = []
        self.decisions = defaultdict(int)
        self.total_count = 0

    def record(self, risk_score: float, decision: str, latency_ms: float):
        """Record a prediction for performance tracking."""
        self.risk_scores.append(risk_score)
        self.latencies.append(latency_ms)
        self.decisions[decision] += 1
        self.total_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        import numpy as np

        if self.total_count == 0:
            return {'error': 'No data recorded'}

        risk_scores = np.array(self.risk_scores)
        latencies = np.array(self.latencies)

        return {
            'total_predictions': self.total_count,
            'risk_score': {
                'mean': float(np.mean(risk_scores)),
                'median': float(np.median(risk_scores)),
                'p95': float(np.percentile(risk_scores, 95)),
                'p99': float(np.percentile(risk_scores, 99)),
                'std': float(np.std(risk_scores))
            },
            'latency_ms': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99))
            },
            'decisions': dict(self.decisions),
            'decision_rates': {
                k: v / self.total_count for k, v in self.decisions.items()
            }
        }

    def publish_summary(self, metrics_publisher: MetricsPublisher):
        """Publish summary statistics to CloudWatch."""
        summary = self.get_summary()

        if 'error' in summary:
            return

        # Risk score metrics
        metrics_publisher.put_metric('RiskScore_Mean', summary['risk_score']['mean'])
        metrics_publisher.put_metric('RiskScore_P95', summary['risk_score']['p95'])
        metrics_publisher.put_metric('RiskScore_P99', summary['risk_score']['p99'])

        # Latency metrics
        metrics_publisher.put_metric('Latency_Mean', summary['latency_ms']['mean'], unit='Milliseconds')
        metrics_publisher.put_metric('Latency_P95', summary['latency_ms']['p95'], unit='Milliseconds')
        metrics_publisher.put_metric('Latency_P99', summary['latency_ms']['p99'], unit='Milliseconds')

        # Decision rates
        for decision, rate in summary['decision_rates'].items():
            metrics_publisher.put_metric(
                f'{decision.capitalize()}Rate',
                rate * 100,
                unit='Percent'
            )

        metrics_publisher.flush()

        print(f"[METRICS] Published performance summary ({self.total_count} predictions)")


class AlertManager:
    """
    Manage CloudWatch alarms for fraud detection system.

    Monitors:
    - High error rate (>1%)
    - High latency (P95 >200ms)
    - Unusual risk score distribution
    - High block rate (>10%)
    """

    def __init__(self, region: str = 'us-east-1'):
        """Initialize alert manager."""
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns = boto3.client('sns', region_name=region)

    def create_alarm(
        self,
        alarm_name: str,
        metric_name: str,
        namespace: str,
        threshold: float,
        comparison: str = 'GreaterThanThreshold',
        evaluation_periods: int = 2,
        period: int = 300,
        statistic: str = 'Average',
        sns_topic_arn: Optional[str] = None
    ):
        """
        Create a CloudWatch alarm.

        Args:
            alarm_name: Name of the alarm
            metric_name: Metric to monitor
            namespace: CloudWatch namespace
            threshold: Alarm threshold
            comparison: Comparison operator
            evaluation_periods: Number of periods to evaluate
            period: Period length in seconds
            statistic: Statistic to use (Average, Sum, etc.)
            sns_topic_arn: Optional SNS topic for notifications
        """
        alarm_params = {
            'AlarmName': alarm_name,
            'ComparisonOperator': comparison,
            'EvaluationPeriods': evaluation_periods,
            'MetricName': metric_name,
            'Namespace': namespace,
            'Period': period,
            'Statistic': statistic,
            'Threshold': threshold,
            'ActionsEnabled': True if sns_topic_arn else False,
            'AlarmDescription': f'Alert when {metric_name} {comparison} {threshold}'
        }

        if sns_topic_arn:
            alarm_params['AlarmActions'] = [sns_topic_arn]

        try:
            self.cloudwatch.put_metric_alarm(**alarm_params)
            print(f"[SUCCESS] Created alarm: {alarm_name}")
        except Exception as e:
            print(f"[ERROR] Failed to create alarm {alarm_name}: {e}")

    def setup_default_alarms(self, namespace: str = 'VPBankFraud', sns_topic_arn: Optional[str] = None):
        """
        Set up default alarms for fraud detection system.

        Args:
            namespace: CloudWatch namespace
            sns_topic_arn: Optional SNS topic for notifications
        """
        print(f"\n[INFO] Setting up default alarms for namespace: {namespace}")

        alarms = [
            # High error rate
            {
                'alarm_name': 'VPBank_HighErrorRate',
                'metric_name': 'ErrorCount',
                'threshold': 10,
                'comparison': 'GreaterThanThreshold',
                'statistic': 'Sum',
                'evaluation_periods': 2,
                'period': 300
            },
            # High latency
            {
                'alarm_name': 'VPBank_HighLatency',
                'metric_name': 'Latency_P95',
                'threshold': 200,
                'comparison': 'GreaterThanThreshold',
                'statistic': 'Average',
                'evaluation_periods': 3,
                'period': 300
            },
            # Unusual risk score (too high)
            {
                'alarm_name': 'VPBank_HighRiskScore',
                'metric_name': 'RiskScore_Mean',
                'threshold': 0.5,
                'comparison': 'GreaterThanThreshold',
                'statistic': 'Average',
                'evaluation_periods': 3,
                'period': 600
            },
            # High block rate
            {
                'alarm_name': 'VPBank_HighBlockRate',
                'metric_name': 'BlockRate',
                'threshold': 10,
                'comparison': 'GreaterThanThreshold',
                'statistic': 'Average',
                'evaluation_periods': 2,
                'period': 600
            }
        ]

        for alarm in alarms:
            self.create_alarm(
                namespace=namespace,
                sns_topic_arn=sns_topic_arn,
                **alarm
            )

        print(f"[SUCCESS] Created {len(alarms)} default alarms")


if __name__ == "__main__":
    """Demo: CloudWatch metrics publishing."""

    print("\n" + "="*60)
    print("CLOUDWATCH METRICS - DEMO")
    print("="*60)

    # Initialize metrics publisher
    metrics = MetricsPublisher(namespace='VPBankFraud')

    # Simulate predictions
    print("\n[INFO] Simulating 10 predictions...")

    import numpy as np
    np.random.seed(42)

    monitor = PerformanceMonitor()

    for i in range(10):
        risk_score = np.random.beta(2, 8)
        model_score = risk_score * 0.7
        entity_risk = risk_score * 0.3
        latency_ms = np.random.gamma(50, 2)

        if risk_score < 0.3:
            decision = 'pass'
        elif risk_score < 0.7:
            decision = 'challenge'
        else:
            decision = 'block'

        # Record metrics
        metrics.record_prediction(
            risk_score=risk_score,
            model_score=model_score,
            entity_risk=entity_risk,
            decision=decision,
            latency_ms=latency_ms,
            channel='online_banking'
        )

        monitor.record(risk_score, decision, latency_ms)

        print(f"  {i+1}. risk={risk_score:.3f}, decision={decision}, latency={latency_ms:.1f}ms")

    # Get summary
    summary = monitor.get_summary()
    print(f"\n[INFO] Performance Summary:")
    print(json.dumps(summary, indent=2))

    # Publish summary metrics
    print(f"\n[INFO] Publishing summary metrics to CloudWatch...")
    monitor.publish_summary(metrics)

    print("\n[SUCCESS] Demo complete!")
    print("\nNote: In production, metrics will be published to CloudWatch.")
    print("      Set AWS credentials to enable actual publishing.")
