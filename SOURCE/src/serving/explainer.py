"""
SHAP-based fraud detection explainability module.

Generates human-readable reason codes from model predictions.
Maps SHAP values to business-friendly explanations.
"""

import numpy as np
import pandas as pd
import shap
import pickle
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FraudExplainer:
    """
    Generate explainable reason codes for fraud predictions.

    Uses SHAP (SHapley Additive exPlanations) to identify top contributing features
    and maps them to human-readable reason codes for business users.
    """

    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize explainer with trained model.

        Args:
            model_path: Path to pickled model file
            scaler_path: Path to scaler (optional, for feature engineering)
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler if provided
        self.scaler = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Initialize SHAP explainer
        # Use LinearExplainer for logistic regression (fast and exact)
        self.explainer = shap.LinearExplainer(self.model, masker=shap.maskers.Independent(data=np.zeros((1, 34))))

        # Feature names (34 features after engineering)
        self.feature_names = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
            'Amount_Log', 'Hour', 'Time_Period_1', 'Time_Period_2'
        ]

        # Reason code mapping (feature → business reason)
        self.reason_code_map = {
            # Amount-related
            'Amount': 'amount_high',
            'Amount_Log': 'amount_unusual',

            # Time-related (fraud often happens at unusual hours)
            'Hour': 'time_unusual',
            'Time': 'time_sequence_anomaly',
            'Time_Period_1': 'time_afternoon_evening',  # Higher fraud in afternoon/evening
            'Time_Period_2': 'time_night',  # Late night transactions

            # PCA features (V1-V28) - map to behavioral patterns
            'V1': 'pattern_anomaly_v1',
            'V2': 'pattern_anomaly_v2',
            'V3': 'pattern_anomaly_v3',
            'V4': 'pattern_behavioral_v4',  # V4 is high importance
            'V5': 'pattern_anomaly_v5',
            'V6': 'pattern_anomaly_v6',
            'V7': 'pattern_anomaly_v7',
            'V8': 'pattern_anomaly_v8',
            'V9': 'pattern_anomaly_v9',
            'V10': 'pattern_behavioral_v10',  # V10 is high importance
            'V11': 'pattern_behavioral_v11',  # V11 is high importance
            'V12': 'pattern_behavioral_v12',  # V12 is high importance
            'V13': 'pattern_anomaly_v13',
            'V14': 'pattern_behavioral_v14',  # V14 is high importance
            'V15': 'pattern_anomaly_v15',
            'V16': 'pattern_anomaly_v16',
            'V17': 'pattern_velocity_v17',  # V17 often relates to velocity
            'V18': 'pattern_anomaly_v18',
            'V19': 'pattern_anomaly_v19',
            'V20': 'pattern_anomaly_v20',
            'V21': 'pattern_anomaly_v21',
            'V22': 'pattern_anomaly_v22',
            'V23': 'pattern_anomaly_v23',
            'V24': 'pattern_anomaly_v24',
            'V25': 'pattern_anomaly_v25',
            'V26': 'pattern_anomaly_v26',
            'V27': 'pattern_anomaly_v27',
            'V28': 'pattern_anomaly_v28',
        }

        # Human-readable descriptions for reason codes
        self.reason_descriptions = {
            'amount_high': 'Transaction amount significantly higher than typical',
            'amount_unusual': 'Unusual transaction amount pattern detected',
            'time_unusual': 'Transaction occurred at unusual hour',
            'time_sequence_anomaly': 'Abnormal time sequence pattern',
            'time_afternoon_evening': 'Transaction during high-risk hours (afternoon/evening)',
            'time_night': 'Late night transaction (higher fraud risk)',
            'pattern_behavioral_v10': 'Unusual behavioral pattern (V10 - top fraud indicator)',
            'pattern_behavioral_v4': 'Unusual behavioral pattern (V4 - high importance)',
            'pattern_behavioral_v11': 'Unusual behavioral pattern (V11 - high importance)',
            'pattern_behavioral_v12': 'Unusual behavioral pattern (V12 - high importance)',
            'pattern_behavioral_v14': 'Unusual behavioral pattern (V14 - high importance)',
            'pattern_velocity_v17': 'Abnormal transaction velocity detected',
            'entity_risk_high': 'Device/IP/Email has history of fraud',
            'sequence_anomaly': 'Autoencoder detected abnormal transaction sequence',
        }

    def explain_prediction(self, features: np.ndarray, entity_risk: float = 0.0,
                          top_k: int = 5) -> Dict:
        """
        Generate explanation for a single prediction.

        Args:
            features: Numpy array of shape (34,) - engineered features
            entity_risk: Entity risk score from Deep Lane (0-1)
            top_k: Number of top reasons to return (default: 5)

        Returns:
            Dictionary with:
                - risk_score: Overall risk score
                - reason_codes: List of top reason codes
                - reason_details: List of detailed explanations
                - shap_values: Raw SHAP values for debugging
        """
        # Ensure 2D array for SHAP
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get model prediction
        risk_score = self.model.predict_proba(features)[0, 1]

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)

        # Get feature importance (absolute SHAP values)
        feature_importance = np.abs(shap_values[0])

        # Get top-k features
        top_indices = np.argsort(feature_importance)[::-1][:top_k]

        # Generate reason codes
        reason_codes = []
        reason_details = []

        for idx in top_indices:
            feature_name = self.feature_names[idx]
            shap_val = shap_values[0][idx]
            feature_val = features[0][idx]

            # Get reason code
            reason_code = self.reason_code_map.get(feature_name, f'pattern_{feature_name.lower()}')

            # Get description
            description = self.reason_descriptions.get(
                reason_code,
                f'Unusual pattern detected in {feature_name}'
            )

            # Add impact direction
            impact = "increases" if shap_val > 0 else "decreases"
            detailed_desc = f"{description} (impact: {impact} fraud risk by {abs(shap_val):.3f})"

            reason_codes.append(reason_code)
            reason_details.append({
                'code': reason_code,
                'description': description,
                'feature': feature_name,
                'feature_value': float(feature_val),
                'shap_value': float(shap_val),
                'impact': impact
            })

        # Add entity risk if significant
        if entity_risk > 0.3:
            reason_codes.insert(0, 'entity_risk_high')
            reason_details.insert(0, {
                'code': 'entity_risk_high',
                'description': self.reason_descriptions['entity_risk_high'],
                'feature': 'entity_risk',
                'feature_value': float(entity_risk),
                'shap_value': float(entity_risk),  # Not SHAP, but for consistency
                'impact': 'increases'
            })

        return {
            'risk_score': float(risk_score),
            'reason_codes': reason_codes[:top_k],
            'reason_details': reason_details[:top_k],
            'shap_values': shap_values[0].tolist(),
            'feature_names': self.feature_names
        }

    def get_feature_importance(self, X: np.ndarray, sample_size: int = 100) -> pd.DataFrame:
        """
        Calculate average feature importance across a sample of transactions.

        Args:
            X: Feature matrix (n_samples, 34)
            sample_size: Number of samples to use (default: 100)

        Returns:
            DataFrame with feature importance rankings
        """
        # Sample if too large
        if X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_sample)

        # Average absolute SHAP values
        avg_importance = np.abs(shap_values).mean(axis=0)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': avg_importance,
            'reason_code': [self.reason_code_map.get(f, f'pattern_{f.lower()}')
                           for f in self.feature_names]
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df


def demo_explainer():
    """
    Demo usage of FraudExplainer.
    """
    import os

    # Paths
    model_path = "models/fast_lane_baseline_20251105_210615/logistic_model.pkl"
    scaler_path = None  # Scaler is inside preprocessor.pkl, but not needed for SHAP

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return

    # Initialize explainer
    print("[INFO] Initializing FraudExplainer...")
    explainer = FraudExplainer(model_path, scaler_path)
    print("[SUCCESS] Explainer ready!\n")

    # Create sample transaction (34 features)
    # This is a high-risk transaction pattern
    sample_features = np.array([
        406,  # Time
        -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09,  # V1-V10
        -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.03, 0.40, 0.25,  # V11-V20
        -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02,  # V21-V28
        500.0,  # Amount (high)
        6.21,  # Amount_Log
        0,  # Hour
        0, 1  # Time_Period (night)
    ])

    # Get explanation
    print("[INFO] Explaining transaction...")
    explanation = explainer.explain_prediction(sample_features, entity_risk=0.42)

    print(f"\n[RESULT] Risk Score: {explanation['risk_score']:.4f}\n")
    print("[REASONS] Top Reasons for Fraud Risk:")
    for i, detail in enumerate(explanation['reason_details'], 1):
        print(f"\n{i}. {detail['code'].upper()}")
        print(f"   {detail['description']}")
        print(f"   Feature: {detail['feature']} = {detail['feature_value']:.2f}")
        print(f"   Impact: {detail['impact']} risk by {abs(detail['shap_value']):.4f}")


if __name__ == "__main__":
    demo_explainer()
