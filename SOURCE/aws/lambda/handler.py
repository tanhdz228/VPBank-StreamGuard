"""
VPBank StreamGuard - Lambda Fraud Scoring Handler
Loads Fast Lane model from S3, fetches entity risk from DynamoDB, scores transactions.
"""

import json
import boto3
import pickle
import numpy as np
import os
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients (outside handler for reuse across invocations)
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Environment variables
MODEL_BUCKET = os.environ.get('MODEL_BUCKET')
MODEL_KEY = os.environ.get('MODEL_KEY', 'fast_lane/lr_model.pkl')
SCALER_KEY = os.environ.get('SCALER_KEY', 'fast_lane/scaler.pkl')
ENTITY_RISK_TABLE = os.environ.get('ENTITY_RISK_TABLE')

# Global variables for caching (persists across warm invocations)
model = None
scaler = None
entity_table = None


def load_model_from_s3():
    """Load model and scaler from S3 (cached in /tmp for warm starts)."""
    global model, scaler

    if model is None:
        logger.info(f"Loading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")

        # Download model to /tmp (Lambda's writable directory)
        model_path = '/tmp/lr_model.pkl'
        s3.download_file(MODEL_BUCKET, MODEL_KEY, model_path)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info("✓ Model loaded and cached")
    else:
        logger.info("Using cached model")

    if scaler is None:
        logger.info(f"Loading scaler from s3://{MODEL_BUCKET}/{SCALER_KEY}")

        scaler_path = '/tmp/scaler.pkl'
        s3.download_file(MODEL_BUCKET, SCALER_KEY, scaler_path)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        logger.info("✓ Scaler loaded and cached")
    else:
        logger.info("Using cached scaler")

    return model, scaler


def get_entity_risk(entity_type: str, entity_id: str) -> float:
    """
    Get entity risk from DynamoDB.

    Args:
        entity_type: Type of entity (e.g., 'id_23', 'DeviceInfo')
        entity_id: Entity identifier

    Returns:
        Risk score (0.0-1.0), or 0.0 if not found
    """
    global entity_table

    if entity_table is None:
        entity_table = dynamodb.Table(ENTITY_RISK_TABLE)

    try:
        # Composite key: type#id (e.g., "id_23#1")
        entity_key = f"{entity_type}#{entity_id}"

        response = entity_table.get_item(
            Key={'entity_key': entity_key},
            ProjectionExpression='risk_score'  # Only fetch what we need
        )

        if 'Item' in response:
            risk_score = float(response['Item']['risk_score'])
            logger.info(f"Entity risk for {entity_key}: {risk_score:.4f}")
            return risk_score
        else:
            logger.info(f"Entity {entity_key} not found, using default 0.0")
            return 0.0

    except Exception as e:
        logger.error(f"Error fetching entity risk for {entity_type}#{entity_id}: {str(e)}")
        return 0.0  # Fail gracefully


def score_transaction(features: np.ndarray) -> float:
    """
    Score transaction with Fast Lane model (Logistic Regression).

    Args:
        features: Feature array (30 features: V1-V28, Time, Amount)

    Returns:
        Risk score (probability of fraud)
    """
    model_obj, scaler_obj = load_model_from_s3()

    # Scale features
    features_scaled = scaler_obj.transform(features)

    # Predict probability of fraud (class 1)
    risk_score = float(model_obj.predict_proba(features_scaled)[0][1])

    logger.info(f"Model risk score: {risk_score:.4f}")
    return risk_score


def apply_rba_policy(combined_score: float) -> str:
    """
    Apply Risk-Based Authentication policy.

    Args:
        combined_score: Combined risk score (0.0-1.0)

    Returns:
        Decision: 'pass', 'challenge', or 'block'
    """
    if combined_score < 0.3:
        return 'pass'
    elif combined_score < 0.7:
        return 'challenge'  # Step-up auth (OTP/biometric)
    else:
        return 'block'


def validate_request(body: Dict[str, Any]) -> bool:
    """Validate request has required fields."""
    # Check V1-V28 present
    for i in range(1, 29):
        if f'V{i}' not in body:
            return False

    # Check Time and Amount
    if 'Time' not in body or 'Amount' not in body:
        return False

    return True


def lambda_handler(event, context):
    """
    Main Lambda handler for fraud scoring.

    Input (JSON):
        {
            "V1": -1.3598, "V2": -0.0727, ..., "V28": 0.0147,
            "Time": 406, "Amount": 100.0,
            "device_id": "device_abc123",  # Optional
            "ip_proxy": "1",               # Optional
            "email_domain": "gmail.com"    # Optional
        }

    Output (JSON):
        {
            "risk_score": 0.4521,
            "model_score": 0.3245,
            "entity_risk": 0.4208,
            "entity_risks": {"device": 0.1, "ip_proxy": 0.4208},
            "decision": "challenge",
            "timestamp": "2025-11-09T12:34:56Z"
        }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    try:
        # Parse request body
        if 'body' in event:
            # API Gateway format
            body = json.loads(event['body'])
        else:
            # Direct invocation or test event
            body = event

        # Validate request
        if not validate_request(body):
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Missing required fields',
                    'message': 'Request must include V1-V28, Time, and Amount'
                })
            }

        # Extract features (V1-V28, Time, Amount) - Total: 30 features
        features = []
        for i in range(1, 29):
            features.append(float(body.get(f'V{i}', 0.0)))
        features.append(float(body.get('Time', 0.0)))
        features.append(float(body.get('Amount', 0.0)))

        features = np.array([features])  # Shape: (1, 30)

        logger.info(f"Features extracted: shape={features.shape}")

        # Get base risk score from Fast Lane model
        model_score = score_transaction(features)

        # Fetch entity risk (if entity identifiers provided)
        entity_risk = 0.0
        entity_risks = {}

        if 'device_id' in body and body['device_id']:
            device_risk = get_entity_risk('DeviceInfo', str(body['device_id']))
            entity_risks['device'] = device_risk
            entity_risk = max(entity_risk, device_risk)

        if 'ip_proxy' in body and body['ip_proxy']:
            ip_risk = get_entity_risk('id_23', str(body['ip_proxy']))
            entity_risks['ip_proxy'] = ip_risk
            entity_risk = max(entity_risk, ip_risk)

        if 'email_domain' in body and body['email_domain']:
            email_risk = get_entity_risk('P_emaildomain', str(body['email_domain']))
            entity_risks['email'] = email_risk
            entity_risk = max(entity_risk, email_risk)

        # Combined score (70% model, 30% entity risk)
        combined_score = 0.7 * model_score + 0.3 * entity_risk

        logger.info(f"Scores - Model: {model_score:.4f}, Entity: {entity_risk:.4f}, Combined: {combined_score:.4f}")

        # Apply RBA policy
        decision = apply_rba_policy(combined_score)

        logger.info(f"Decision: {decision}")

        # Build response
        response_body = {
            'risk_score': round(combined_score, 4),
            'model_score': round(model_score, 4),
            'entity_risk': round(entity_risk, 4),
            'entity_risks': entity_risks,
            'decision': decision,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'X-Request-Id': context.request_id if context else 'test'
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error',
                'request_id': context.request_id if context else 'test'
            })
        }


def health_handler(event, context):
    """Health check endpoint."""
    try:
        # Check if model can be loaded
        load_model_from_s3()

        # Check DynamoDB connection
        global entity_table
        if entity_table is None:
            entity_table = dynamodb.Table(ENTITY_RISK_TABLE)

        # Simple query to verify table access
        entity_table.get_item(Key={'entity_key': 'health_check'})

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'healthy',
                'model_loaded': model is not None,
                'dynamodb_connected': True,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'statusCode': 503,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e)
            })
        }
