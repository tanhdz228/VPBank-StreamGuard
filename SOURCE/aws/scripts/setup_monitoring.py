"""
Setup CloudWatch Monitoring for VPBank StreamGuard

Creates:
- CloudWatch alarms for error rate, latency, unusual patterns
- CloudWatch dashboard for real-time monitoring
- SNS topic for alarm notifications (optional)
- Billing alarm

Usage:
    python aws/scripts/setup_monitoring.py --email your.email@vpbank.com.vn
"""

import boto3
import argparse
import json
from typing import Optional


class MonitoringSetup:
    """Setup CloudWatch monitoring infrastructure for fraud detection API."""

    def __init__(self, region: str = 'us-east-1', stack_name: str = 'vpbank-fraud-detection'):
        """
        Initialize monitoring setup.

        Args:
            region: AWS region
            stack_name: CloudFormation stack name (for Lambda function name)
        """
        self.region = region
        self.stack_name = stack_name
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        self.lambda_client = boto3.client('lambda', region_name=region)

        # Get Lambda function name from stack
        self.lambda_function_name = 'vpbank-fraud-scoring'
        self.namespace = 'VPBankFraud'

    def create_sns_topic(self, email: str) -> str:
        """
        Create SNS topic for alarm notifications.

        Args:
            email: Email address for notifications

        Returns:
            SNS topic ARN
        """
        topic_name = 'VPBankFraudAlerts'

        print(f"\n[INFO] Creating SNS topic: {topic_name}")

        try:
            response = self.sns.create_topic(Name=topic_name)
            topic_arn = response['TopicArn']

            # Subscribe email
            self.sns.subscribe(
                TopicArn=topic_arn,
                Protocol='email',
                Endpoint=email
            )

            print(f"[SUCCESS] SNS topic created: {topic_arn}")
            print(f"[INFO] Subscription confirmation sent to {email}")
            print(f"[ACTION] Please check your email and confirm the subscription!")

            return topic_arn

        except Exception as e:
            print(f"[ERROR] Failed to create SNS topic: {e}")
            return None

    def create_alarms(self, sns_topic_arn: Optional[str] = None):
        """
        Create CloudWatch alarms for fraud detection system.

        Args:
            sns_topic_arn: Optional SNS topic ARN for notifications
        """
        print(f"\n[INFO] Creating CloudWatch alarms...")

        alarms = [
            # Lambda errors
            {
                'AlarmName': 'VPBank_LambdaErrors',
                'MetricName': 'Errors',
                'Namespace': 'AWS/Lambda',
                'Dimensions': [{'Name': 'FunctionName', 'Value': self.lambda_function_name}],
                'Statistic': 'Sum',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 5,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when Lambda function has >5 errors in 5 minutes'
            },
            # Lambda throttles
            {
                'AlarmName': 'VPBank_LambdaThrottles',
                'MetricName': 'Throttles',
                'Namespace': 'AWS/Lambda',
                'Dimensions': [{'Name': 'FunctionName', 'Value': self.lambda_function_name}],
                'Statistic': 'Sum',
                'Period': 300,
                'EvaluationPeriods': 1,
                'Threshold': 1,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when Lambda function is throttled'
            },
            # Lambda duration (P95)
            {
                'AlarmName': 'VPBank_LambdaHighDuration',
                'MetricName': 'Duration',
                'Namespace': 'AWS/Lambda',
                'Dimensions': [{'Name': 'FunctionName', 'Value': self.lambda_function_name}],
                'ExtendedStatistic': 'p95',
                'Period': 300,
                'EvaluationPeriods': 3,
                'Threshold': 200,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when P95 Lambda duration >200ms'
            },
            # API Gateway 5xx errors
            {
                'AlarmName': 'VPBank_API5xxErrors',
                'MetricName': '5XXError',
                'Namespace': 'AWS/ApiGateway',
                'Dimensions': [{'Name': 'ApiName', 'Value': 'vpbank-fraud-api'}],
                'Statistic': 'Sum',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 5,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when API Gateway has >5 5xx errors in 5 minutes'
            },
            # API Gateway 4xx errors (high rate may indicate attack)
            {
                'AlarmName': 'VPBank_API4xxErrors',
                'MetricName': '4XXError',
                'Namespace': 'AWS/ApiGateway',
                'Dimensions': [{'Name': 'ApiName', 'Value': 'vpbank-fraud-api'}],
                'Statistic': 'Sum',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 50,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when API Gateway has >50 4xx errors in 5 minutes'
            },
            # DynamoDB throttles
            {
                'AlarmName': 'VPBank_DynamoDBThrottles',
                'MetricName': 'UserErrors',
                'Namespace': 'AWS/DynamoDB',
                'Dimensions': [{'Name': 'TableName', 'Value': 'vpbank-entity-risk'}],
                'Statistic': 'Sum',
                'Period': 300,
                'EvaluationPeriods': 2,
                'Threshold': 5,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': 'Alert when DynamoDB has throttling errors'
            }
        ]

        # Add SNS action if topic provided
        if sns_topic_arn:
            for alarm in alarms:
                alarm['AlarmActions'] = [sns_topic_arn]
                alarm['OKActions'] = [sns_topic_arn]  # Notify when alarm clears
                alarm['ActionsEnabled'] = True

        # Create alarms
        created_count = 0
        for alarm in alarms:
            try:
                # Handle ExtendedStatistic vs Statistic
                if 'ExtendedStatistic' in alarm:
                    statistic_param = {'ExtendedStatistic': alarm.pop('ExtendedStatistic')}
                else:
                    statistic_param = {'Statistic': alarm.pop('Statistic')}

                self.cloudwatch.put_metric_alarm(
                    **alarm,
                    **statistic_param
                )
                print(f"  ✓ Created: {alarm['AlarmName']}")
                created_count += 1
            except Exception as e:
                print(f"  ✗ Failed: {alarm['AlarmName']} - {e}")

        print(f"\n[SUCCESS] Created {created_count}/{len(alarms)} alarms")

    def create_billing_alarm(self, threshold: float = 10.0, sns_topic_arn: Optional[str] = None):
        """
        Create billing alarm to prevent cost overruns.

        Args:
            threshold: Billing threshold in USD
            sns_topic_arn: Optional SNS topic ARN for notifications
        """
        print(f"\n[INFO] Creating billing alarm (threshold: ${threshold})...")

        try:
            alarm_params = {
                'AlarmName': 'VPBank_BillingAlert',
                'MetricName': 'EstimatedCharges',
                'Namespace': 'AWS/Billing',
                'Statistic': 'Maximum',
                'Period': 21600,  # 6 hours
                'EvaluationPeriods': 1,
                'Threshold': threshold,
                'ComparisonOperator': 'GreaterThanThreshold',
                'AlarmDescription': f'Alert when estimated charges exceed ${threshold}',
                'Dimensions': [{'Name': 'Currency', 'Value': 'USD'}]
            }

            if sns_topic_arn:
                alarm_params['AlarmActions'] = [sns_topic_arn]
                alarm_params['ActionsEnabled'] = True

            # Note: Billing metrics are only available in us-east-1
            billing_cloudwatch = boto3.client('cloudwatch', region_name='us-east-1')
            billing_cloudwatch.put_metric_alarm(**alarm_params)

            print(f"[SUCCESS] Billing alarm created (threshold: ${threshold})")
            print(f"[INFO] Billing metrics are checked every 6 hours")

        except Exception as e:
            print(f"[ERROR] Failed to create billing alarm: {e}")
            print(f"[INFO] Make sure billing metrics are enabled in CloudWatch console")

    def create_dashboard(self):
        """Create CloudWatch dashboard for fraud detection monitoring."""

        print(f"\n[INFO] Creating CloudWatch dashboard...")

        dashboard_body = {
            "widgets": [
                # Lambda Invocations
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Lambda", "Invocations", {"stat": "Sum", "label": "Invocations"}],
                            [".", "Errors", {"stat": "Sum", "label": "Errors"}],
                            [".", "Throttles", {"stat": "Sum", "label": "Throttles"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Lambda Invocations & Errors",
                        "period": 300,
                        "dimensions": {"FunctionName": self.lambda_function_name}
                    }
                },
                # Lambda Duration
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Lambda", "Duration", {"stat": "Average", "label": "Avg"}],
                            ["...", {"stat": "p95", "label": "P95"}],
                            ["...", {"stat": "p99", "label": "P99"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Lambda Duration (ms)",
                        "period": 300,
                        "dimensions": {"FunctionName": self.lambda_function_name}
                    }
                },
                # API Gateway Requests
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/ApiGateway", "Count", {"stat": "Sum", "label": "Requests"}],
                            [".", "4XXError", {"stat": "Sum", "label": "4xx Errors"}],
                            [".", "5XXError", {"stat": "Sum", "label": "5xx Errors"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "API Gateway Requests & Errors",
                        "period": 300
                    }
                },
                # DynamoDB Metrics
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/DynamoDB", "ConsumedReadCapacityUnits", {"stat": "Sum"}],
                            [".", "UserErrors", {"stat": "Sum"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "DynamoDB Activity",
                        "period": 300,
                        "dimensions": {"TableName": "vpbank-entity-risk"}
                    }
                },
                # Custom Metrics: Risk Score Distribution
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "RiskScore_Mean", {"stat": "Average", "label": "Mean"}],
                            [".", "RiskScore_P95", {"stat": "Average", "label": "P95"}],
                            [".", "RiskScore_P99", {"stat": "Average", "label": "P99"}]
                        ],
                        "view": "timeSeries",
                        "stacked": False,
                        "region": self.region,
                        "title": "Risk Score Distribution",
                        "period": 300,
                        "yAxis": {"left": {"min": 0, "max": 1}}
                    }
                },
                # Custom Metrics: Decision Rates
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            [self.namespace, "PassRate", {"stat": "Average", "label": "Pass %"}],
                            [".", "ChallengeRate", {"stat": "Average", "label": "Challenge %"}],
                            [".", "BlockRate", {"stat": "Average", "label": "Block %"}]
                        ],
                        "view": "timeSeries",
                        "stacked": True,
                        "region": self.region,
                        "title": "Decision Distribution",
                        "period": 300,
                        "yAxis": {"left": {"min": 0, "max": 100}}
                    }
                }
            ]
        }

        try:
            self.cloudwatch.put_dashboard(
                DashboardName='VPBankFraudDetection',
                DashboardBody=json.dumps(dashboard_body)
            )

            dashboard_url = f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name=VPBankFraudDetection"

            print(f"[SUCCESS] Dashboard created!")
            print(f"[INFO] View dashboard at:")
            print(f"       {dashboard_url}")

        except Exception as e:
            print(f"[ERROR] Failed to create dashboard: {e}")

    def setup_all(self, email: Optional[str] = None, billing_threshold: float = 10.0):
        """
        Setup all monitoring components.

        Args:
            email: Email for SNS notifications (optional)
            billing_threshold: Billing alert threshold in USD
        """
        print("\n" + "="*60)
        print("CLOUDWATCH MONITORING SETUP")
        print("="*60)

        # Create SNS topic if email provided
        sns_topic_arn = None
        if email:
            sns_topic_arn = self.create_sns_topic(email)

        # Create alarms
        self.create_alarms(sns_topic_arn)

        # Create billing alarm
        self.create_billing_alarm(billing_threshold, sns_topic_arn)

        # Create dashboard
        self.create_dashboard()

        print("\n" + "="*60)
        print("MONITORING SETUP COMPLETE")
        print("="*60)

        if sns_topic_arn:
            print(f"\n✅ SNS Topic: {sns_topic_arn}")
            print(f"   → Check email {email} and confirm subscription!")

        print(f"\n✅ CloudWatch Alarms: 7 alarms created")
        print(f"   → View at: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#alarmsV2:")

        print(f"\n✅ CloudWatch Dashboard: VPBankFraudDetection")
        print(f"   → View at: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name=VPBankFraudDetection")

        print(f"\n💡 Next Steps:")
        print(f"   1. Confirm SNS email subscription (if provided)")
        print(f"   2. Review alarms in CloudWatch console")
        print(f"   3. Customize dashboard as needed")
        print(f"   4. Test alarms by triggering conditions")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Setup CloudWatch monitoring for VPBank fraud detection API'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Email address for alarm notifications (optional)'
    )
    parser.add_argument(
        '--billing-threshold',
        type=float,
        default=10.0,
        help='Billing alert threshold in USD (default: 10.0)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )

    args = parser.parse_args()

    # Setup monitoring
    setup = MonitoringSetup(region=args.region)
    setup.setup_all(
        email=args.email,
        billing_threshold=args.billing_threshold
    )


if __name__ == "__main__":
    main()
