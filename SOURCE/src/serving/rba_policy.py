"""
Risk-Based Authentication (RBA) Policy Engine.

Determines authentication requirements based on risk scores,
channel, time-of-day, and user history.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np


class RBAPolicy:
    """
    Risk-Based Authentication policy engine.

    Maps risk scores to authentication decisions:
    - Pass: Low risk, allow transaction immediately
    - Challenge: Medium risk, require step-up authentication (OTP/biometric)
    - Block: High risk, block transaction and flag for investigation
    """

    def __init__(self,
                 base_thresholds: Optional[Dict[str, float]] = None,
                 channel_adjustments: Optional[Dict[str, float]] = None,
                 time_adjustments: Optional[Dict[str, float]] = None):
        """
        Initialize RBA policy engine.

        Args:
            base_thresholds: Base risk score thresholds
                Default: {'pass': 0.3, 'challenge': 0.7}
            channel_adjustments: Risk threshold adjustments by channel
                Positive = more lenient, Negative = more strict
            time_adjustments: Risk threshold adjustments by hour
        """
        # Base thresholds (score < pass → allow, pass-challenge → step-up, >challenge → block)
        self.base_thresholds = base_thresholds or {
            'pass': 0.3,        # Allow if risk < 0.3
            'challenge': 0.7     # Challenge if 0.3 <= risk < 0.7, Block if >= 0.7
        }

        # Channel-specific adjustments
        # online_banking is baseline (0), others are relative adjustments
        self.channel_adjustments = channel_adjustments or {
            'online_banking': 0.0,     # Baseline (no adjustment)
            'mobile_app': -0.05,       # Slightly more strict (trusted app)
            'atm': 0.05,               # Slightly more lenient (physical card present)
            'pos': 0.03,               # Physical transaction (card present)
            'card_not_present': 0.10,  # More lenient (higher baseline risk)
            'unknown': 0.02            # Slight adjustment for unknown channel
        }

        # Time-of-day adjustments (hour 0-23)
        # Peak hours (9am-5pm) = more lenient (normal activity)
        # Off-hours (late night, early morning) = more strict (suspicious)
        self.time_adjustments = time_adjustments or {
            **{h: 0.10 for h in range(0, 6)},   # 12am-6am: Strict (night, +10% threshold)
            **{h: 0.05 for h in range(6, 9)},   # 6am-9am: Slightly strict (early morning)
            **{h: 0.0 for h in range(9, 18)},   # 9am-6pm: Baseline (business hours)
            **{h: 0.03 for h in range(18, 22)}, # 6pm-10pm: Slightly strict (evening)
            **{h: 0.08 for h in range(22, 24)}  # 10pm-12am: Strict (late night)
        }

        # User trust level adjustments (based on history)
        self.trust_adjustments = {
            'new_user': 0.0,          # No adjustment for new users
            'trusted': -0.10,          # More lenient for trusted users (-10%)
            'suspicious': 0.15,        # More strict for suspicious users (+15%)
            'high_value': -0.05        # Slightly more lenient for high-value customers
        }

    def get_adjusted_thresholds(self,
                                 channel: str = 'online_banking',
                                 hour: Optional[int] = None,
                                 trust_level: str = 'new_user') -> Dict[str, float]:
        """
        Calculate adjusted thresholds based on context.

        Args:
            channel: Transaction channel
            hour: Hour of day (0-23), uses current hour if None
            trust_level: User trust level

        Returns:
            Dictionary with adjusted thresholds: {'pass': float, 'challenge': float}
        """
        # Start with base thresholds
        pass_threshold = self.base_thresholds['pass']
        challenge_threshold = self.base_thresholds['challenge']

        # Apply channel adjustment
        channel_adj = self.channel_adjustments.get(channel, 0.0)

        # Apply time adjustment
        if hour is None:
            hour = datetime.now().hour
        time_adj = self.time_adjustments.get(hour, 0.0)

        # Apply trust level adjustment
        trust_adj = self.trust_adjustments.get(trust_level, 0.0)

        # Total adjustment (additive)
        total_adj = channel_adj + time_adj + trust_adj

        # Apply adjustments (increase threshold = more lenient)
        adjusted_pass = max(0.0, min(1.0, pass_threshold + total_adj))
        adjusted_challenge = max(0.0, min(1.0, challenge_threshold + total_adj))

        # Ensure pass < challenge
        if adjusted_pass >= adjusted_challenge:
            adjusted_challenge = min(1.0, adjusted_pass + 0.1)

        return {
            'pass': adjusted_pass,
            'challenge': adjusted_challenge,
            'adjustments': {
                'channel': channel_adj,
                'time': time_adj,
                'trust': trust_adj,
                'total': total_adj
            }
        }

    def make_decision(self,
                      risk_score: float,
                      channel: str = 'online_banking',
                      hour: Optional[int] = None,
                      trust_level: str = 'new_user',
                      reason_codes: Optional[list] = None) -> Dict:
        """
        Make authentication decision based on risk score and context.

        Args:
            risk_score: Fraud risk score (0-1)
            channel: Transaction channel
            hour: Hour of day (0-23)
            trust_level: User trust level
            reason_codes: List of reason codes explaining the risk

        Returns:
            Dictionary with:
                - decision: 'pass', 'challenge', or 'block'
                - risk_score: Original risk score
                - thresholds: Adjusted thresholds used
                - action: Recommended action
                - message: User-facing message
        """
        # Get adjusted thresholds
        thresholds = self.get_adjusted_thresholds(channel, hour, trust_level)

        # Make decision
        if risk_score < thresholds['pass']:
            decision = 'pass'
            action = 'allow'
            message = 'Transaction approved'
            color = 'green'
        elif risk_score < thresholds['challenge']:
            decision = 'challenge'
            action = 'step_up_auth'
            message = 'Additional verification required. Please confirm with OTP or biometric.'
            color = 'yellow'
        else:
            decision = 'block'
            action = 'block_and_review'
            message = 'Transaction blocked for security review. Please contact support.'
            color = 'red'

        # Build response
        response = {
            'decision': decision,
            'action': action,
            'message': message,
            'color': color,
            'risk_score': float(risk_score),
            'thresholds': {
                'pass': thresholds['pass'],
                'challenge': thresholds['challenge']
            },
            'context': {
                'channel': channel,
                'hour': hour if hour is not None else datetime.now().hour,
                'trust_level': trust_level
            },
            'adjustments': thresholds['adjustments']
        }

        # Add reason codes if provided
        if reason_codes:
            response['reason_codes'] = reason_codes
            response['primary_reason'] = reason_codes[0] if reason_codes else None

        return response

    def get_recommended_actions(self, decision: str) -> Dict[str, any]:
        """
        Get recommended actions for each decision type.

        Args:
            decision: Decision type ('pass', 'challenge', 'block')

        Returns:
            Dictionary with recommended actions and details
        """
        actions = {
            'pass': {
                'user_action': 'none',
                'system_action': 'approve_transaction',
                'monitoring': 'log_for_analysis',
                'user_message': 'Transaction approved',
                'internal_note': 'Low risk - approved automatically'
            },
            'challenge': {
                'user_action': 'verify_identity',
                'system_action': 'request_step_up_auth',
                'monitoring': 'track_auth_response',
                'auth_methods': ['otp_sms', 'otp_email', 'biometric', 'security_question'],
                'timeout': 300,  # 5 minutes to respond
                'user_message': 'Please verify your identity to continue',
                'internal_note': 'Medium risk - requires additional verification'
            },
            'block': {
                'user_action': 'contact_support',
                'system_action': 'block_transaction',
                'monitoring': 'create_fraud_case',
                'escalation': 'fraud_team_review',
                'user_message': 'Transaction blocked. Please contact customer support.',
                'internal_note': 'High risk - blocked for manual review'
            }
        }

        return actions.get(decision, actions['block'])  # Default to block if unknown


def demo_rba_policy():
    """
    Demo usage of RBA policy engine.
    """
    # Initialize policy engine
    print("[INFO] Initializing RBA Policy Engine...\n")
    rba = RBAPolicy()

    # Test scenarios
    scenarios = [
        {
            'name': 'Low Risk - Business Hours',
            'risk_score': 0.15,
            'channel': 'mobile_app',
            'hour': 14,  # 2pm
            'trust_level': 'trusted'
        },
        {
            'name': 'Medium Risk - Evening',
            'risk_score': 0.45,
            'channel': 'online_banking',
            'hour': 20,  # 8pm
            'trust_level': 'new_user'
        },
        {
            'name': 'High Risk - Late Night',
            'risk_score': 0.75,
            'channel': 'card_not_present',
            'hour': 2,  # 2am
            'trust_level': 'new_user'
        },
        {
            'name': 'Borderline - Trusted User',
            'risk_score': 0.35,
            'channel': 'online_banking',
            'hour': 10,  # 10am
            'trust_level': 'trusted'
        }
    ]

    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")

        decision = rba.make_decision(
            risk_score=scenario['risk_score'],
            channel=scenario['channel'],
            hour=scenario['hour'],
            trust_level=scenario['trust_level'],
            reason_codes=['amount_high', 'time_unusual']
        )

        print(f"Risk Score: {decision['risk_score']:.3f}")
        print(f"Channel: {decision['context']['channel']}")
        print(f"Hour: {decision['context']['hour']}:00")
        print(f"Trust Level: {decision['context']['trust_level']}")
        print(f"\nThresholds (adjusted):")
        print(f"  Pass: < {decision['thresholds']['pass']:.3f}")
        print(f"  Challenge: {decision['thresholds']['pass']:.3f} - {decision['thresholds']['challenge']:.3f}")
        print(f"  Block: >= {decision['thresholds']['challenge']:.3f}")
        print(f"\nAdjustments:")
        for key, value in decision['adjustments'].items():
            print(f"  {key}: {value:+.3f}")
        print(f"\n[DECISION] {decision['decision'].upper()} ({decision['color']})")
        print(f"Action: {decision['action']}")
        print(f"Message: {decision['message']}")

        # Get recommended actions
        actions = rba.get_recommended_actions(decision['decision'])
        print(f"\nRecommended Actions:")
        print(f"  User: {actions['user_action']}")
        print(f"  System: {actions['system_action']}")
        print(f"  Monitoring: {actions['monitoring']}")

    print(f"\n{'='*60}")
    print("[SUCCESS] RBA Policy Demo Complete!")


if __name__ == "__main__":
    demo_rba_policy()
