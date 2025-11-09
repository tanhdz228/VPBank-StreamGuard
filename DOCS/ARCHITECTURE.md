# VPBank StreamGuard - AWS Serverless Architecture
**Budget:** $100 credit
**Estimated Monthly Cost:** $1-5 (20+ months runway)
**Architecture:** Serverless (DynamoDB + Lambda + API Gateway)
---
## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Client Application                              │
│              (Web / Mobile / Backend Service)                        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           │ HTTPS POST /predict
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     API Gateway (REST)                               │
│  • Rate limiting: 1000 req/sec                                       │
│  • Authentication: API Key                                           │
│  • CORS enabled                                                      │
│  • Logging to CloudWatch                                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           │ Invoke
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│               Lambda Function (Python 3.11)                          │
│                                                                      │
│  Processing Steps:                                                   │
│  1. Parse request (V1-V28, Time, Amount, entity IDs)                │
│  2. Load Fast Lane model from S3 (cached in /tmp)                   │
│  3. Fetch entity_risk from DynamoDB (batch get)                     │
│  4. Score transaction with Logistic Regression                      │
│  5. Combine: 0.7 * model_score + 0.3 * entity_risk                  │
│  6. Apply RBA rules (pass/challenge/block)                          │
│  7. Return JSON response                                            │
│                                                                      │
│  Configuration:                                                      │
│  • Memory: 512 MB                                                    │
│  • Timeout: 10 seconds                                               │
│  • Runtime: Python 3.11                                              │
└─────────┬────────────────────────────────────┬───────────────────────┘
          │                                    │
          │ GetObject                          │ GetItem
          │ (once, cached)                     │ (per request)
          ▼                                    ▼
┌─────────────────────────┐      ┌───────────────────────────────────┐
│      S3 Bucket          │      │      DynamoDB Table               │
│                         │      │                                   │
│  Models:                │      │  Table: entity_risk               │
│  • Fast Lane model      │      │  Partition Key: entity_key        │
│    (logistic_model.pkl) │      │                                   │
│  • Preprocessor         │      │  Attributes:                      │
│    (preprocessor.pkl)   │      │  • entity_type (str)              │
│                         │      │  • entity_id (str)                │
│  Size: ~500 MB          │      │  • risk_score (float)             │
│  Cost: $0.01/month      │      │  • fraud_rate (float)             │
│                         │      │  • transaction_count (int)        │
│                         │      │                                   │
│                         │      │  Capacity:                        │
│                         │      │  • Items: 5,526                   │
│                         │      │  • Size: ~500 KB                  │
│                         │      │  • RCU: 5 (free tier: 25)         │
│                         │      │  • WCU: 1 (free tier: 25)         │
│                         │      │  • Cost: FREE (within tier)       │
└─────────┬───────────────┘      └───────────────────────────────────┘
          │
          │ PutObject (deployment)
          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  CloudWatch Logs & Metrics                           │
│                                                                      │
│  • Lambda execution logs                                             │
│  • API Gateway access logs                                           │
│  • DynamoDB metrics                                                  │
│  • Custom metrics: latency, fraud_rate, decision_distribution        │
│                                                                      │
│  Cost: FREE (within 5GB tier)                                        │
└──────────────────────────────────────────────────────────────────────┘
```
---
## Cost Breakdown
### Free Tier Limits (12 months)

- **Lambda:** 1M requests/month + 400,000 GB-seconds compute
- **API Gateway:** 1M requests/month
- **DynamoDB:** 25 GB storage + 25 RCU + 25 WCU
- **S3:** 5 GB storage + 20,000 GET + 2,000 PUT
- **CloudWatch:** 5 GB logs
### Estimated Usage (Prototype Testing)

- **Requests:** 10,000/day = 300,000/month
- **Lambda executions:** 300K/month @ 512MB, 500ms avg
- **DynamoDB reads:** 300K/month (1 read per request)
- **S3 storage:** 500 MB models
- **CloudWatch logs:** 1 GB/month
### Monthly Cost Estimate

| Service | Usage | Free Tier | Billable | Cost/Month |
|--------------|-------------------|-----------|----------|------------|
| Lambda | 300K req, 150K GB-s | 1M req | $0 | **$0.00** |
| API Gateway | 300K requests | 1M req | $0 | **$0.00** |
| DynamoDB | 5,526 items, 500KB | 25 RCU | $0 | **$0.00** |
| S3 Storage | 500 MB | 5 GB | $0 | **$0.00** |
| S3 Requests | 300K GET | 20K GET | 280K | **$0.11** |
| CloudWatch | 1 GB logs | 5 GB | $0 | **$0.00** |
| **TOTAL** | | | | **~$0.11** |
**With $100 credit:** 909 months runtime (75+ years!) 
### Production Scale (100,000 req/day = 3M/month)

| Service | Billable | Cost/Month |
|--------------|----------------|------------|
| Lambda | 2M req | $0.40 |
| API Gateway | 2M req | $7.00 |
| DynamoDB | Within tier | $0.00 |
| S3 | 3M GET | $1.20 |
| CloudWatch | 5 GB logs | $2.50 |
| **TOTAL** | | **~$11.10**|
**With $100 credit:** 9 months at production scale
---
## Deployment Architecture
### 1. Infrastructure as Code (AWS SAM)

**File:** `template.yaml` (AWS SAM template)
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: VPBank StreamGuard - Serverless Fraud Detection API
Globals:
Function:
Timeout: 10
MemorySize: 512
Runtime: python3.11
Resources:
# DynamoDB Table
EntityRiskTable:
Type: AWS::DynamoDB::Table
Properties:
TableName: vpbank-entity-risk
BillingMode: PAY_PER_REQUEST # Auto-scaling, no provisioned capacity
AttributeDefinitions:
- AttributeName: entity_key
AttributeType: S
KeySchema:
- AttributeName: entity_key
KeyType: HASH
Tags:
- Key: Project
Value: VPBank-StreamGuard
# S3 Bucket for Models
ModelBucket:
Type: AWS::S3::Bucket
Properties:
BucketName: vpbank-fraud-models
VersioningConfiguration:
Status: Enabled
LifecycleConfiguration:
Rules:
- Id: DeleteOldVersions
Status: Enabled
NoncurrentVersionExpirationInDays: 30
# Lambda Function
FraudScoringFunction:
Type: AWS::Serverless::Function
Properties:
FunctionName: vpbank-fraud-scoring
CodeUri: lambda/
Handler: handler.lambda_handler
Environment:
Variables:
MODEL_BUCKET: !Ref ModelBucket
MODEL_KEY: fast_lane/lr_model.pkl
SCALER_KEY: fast_lane/scaler.pkl
ENTITY_RISK_TABLE: !Ref EntityRiskTable
Policies:
- S3ReadPolicy:
BucketName: !Ref ModelBucket
- DynamoDBReadPolicy:
TableName: !Ref EntityRiskTable
Events:
PredictAPI:
Type: Api
Properties:
Path: /predict
Method: post
RestApiId: !Ref FraudAPI
# API Gateway
FraudAPI:
Type: AWS::Serverless::Api
Properties:
Name: vpbank-fraud-api
StageName: prod
Cors:
AllowMethods: "'POST, GET, OPTIONS'"
AllowHeaders: "'Content-Type,X-Api-Key'"
AllowOrigin: "'*'"
Auth:
ApiKeyRequired: true
DefinitionBody:
openapi: 3.0.1
info:
title: VPBank StreamGuard API
version: 1.0.0
paths:
/predict:
post:
summary: Score transaction for fraud
requestBody:
required: true
content:
application/json:
schema:
type: object
responses:
'200':
description: Successful scoring
/health:
get:
summary: Health check
responses:
'200':
description: Service healthy
Outputs:
ApiUrl:
Description: API Gateway endpoint URL
Value: !Sub "https://${FraudAPI}.execute-api.${AWS::Region}.amazonaws.com/prod"
FunctionArn:
Description: Lambda function ARN
Value: !GetAtt FraudScoringFunction.Arn
TableName:
Description: DynamoDB table name
Value: !Ref EntityRiskTable
```
---
## Lambda Function Implementation
### Directory Structure

```
lambda/
handler.py # Main Lambda handler
requirements.txt # Dependencies
utils/
__init__.py
model_loader.py # S3 model loading with caching
entity_risk.py # DynamoDB entity risk lookup
scoring.py # Fraud scoring logic
```
### handler.py (Main Lambda Function)

```python
import json
import boto3
import pickle
import numpy as np
import os
from typing import Dict, Any
from datetime import datetime
# Initialize AWS clients (outside handler for reuse)
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
# Environment variables
MODEL_BUCKET = os.environ['MODEL_BUCKET']
MODEL_KEY = os.environ['MODEL_KEY']
SCALER_KEY = os.environ['SCALER_KEY']
ENTITY_RISK_TABLE = os.environ['ENTITY_RISK_TABLE']
# Global variables for caching
model = None
scaler = None
entity_table = None
def load_model_from_s3():
"""Load model from S3 (cached in /tmp)."""
global model, scaler
if model is None:
print("Loading model from S3...")
# Download model to /tmp
model_path = '/tmp/lr_model.pkl'
s3.download_file(MODEL_BUCKET, MODEL_KEY, model_path)
with open(model_path, 'rb') as f:
model = pickle.load(f)
print(" Model loaded")
if scaler is None:
print("Loading scaler from S3...")
scaler_path = '/tmp/scaler.pkl'
s3.download_file(MODEL_BUCKET, SCALER_KEY, scaler_path)
with open(scaler_path, 'rb') as f:
scaler = pickle.load(f)
print(" Scaler loaded")
return model, scaler
def get_entity_risk(entity_type: str, entity_id: str) -> float:
"""Get entity risk from DynamoDB."""
global entity_table
if entity_table is None:
entity_table = dynamodb.Table(ENTITY_RISK_TABLE)
try:
# Composite key: type#id
entity_key = f"{entity_type}#{entity_id}"
response = entity_table.get_item(
Key={'entity_key': entity_key}
)
if 'Item' in response:
return float(response['Item']['risk_score'])
else:
return 0.0 # Default for unknown entities
except Exception as e:
print(f"Error fetching entity risk: {e}")
return 0.0
def score_transaction(features: np.ndarray) -> float:
"""Score transaction with Fast Lane model."""
model_obj, scaler_obj = load_model_from_s3()
# Scale features
features_scaled = scaler_obj.transform(features)
# Predict probability of fraud (class 1)
risk_score = float(model_obj.predict_proba(features_scaled)[0][1])
return risk_score
def apply_rba_policy(combined_score: float) -> str:
"""Apply risk-based authentication policy."""
if combined_score < 0.3:
return 'pass'
elif combined_score < 0.7:
return 'challenge' # Step-up auth (OTP/biometric)
else:
return 'block'
def lambda_handler(event, context):
"""Main Lambda handler."""
try:
# Parse request
if 'body' in event:
body = json.loads(event['body'])
else:
body = event
# Extract features (V1-V28, Time, Amount)
features = []
for i in range(1, 29):
features.append(float(body.get(f'V{i}', 0.0)))
features.append(float(body.get('Time', 0.0)))
features.append(float(body.get('Amount', 0.0)))
features = np.array([features])
# Get base risk score from model
model_score = score_transaction(features)
# Fetch entity risk (if available)
entity_risk = 0.0
entity_risks = {}
if 'device_id' in body and body['device_id']:
device_risk = get_entity_risk('DeviceInfo', body['device_id'])
entity_risks['device'] = device_risk
entity_risk = max(entity_risk, device_risk)
if 'ip_proxy' in body and body['ip_proxy']:
ip_risk = get_entity_risk('id_23', body['ip_proxy'])
entity_risks['ip_proxy'] = ip_risk
entity_risk = max(entity_risk, ip_risk)
if 'email_domain' in body and body['email_domain']:
email_risk = get_entity_risk('P_emaildomain', body['email_domain'])
entity_risks['email'] = email_risk
entity_risk = max(entity_risk, email_risk)
# Combined score (70% model, 30% entity risk)
combined_score = 0.7 * model_score + 0.3 * entity_risk
# Apply RBA policy
decision = apply_rba_policy(combined_score)
# Build response
response = {
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
'Access-Control-Allow-Origin': '*'
},
'body': json.dumps(response)
}
except Exception as e:
print(f"Error: {str(e)}")
return {
'statusCode': 500,
'headers': {
'Content-Type': 'application/json',
'Access-Control-Allow-Origin': '*'
},
'body': json.dumps({
'error': str(e),
'message': 'Internal server error'
})
}
```
### requirements.txt

```
boto3==1.34.0
numpy==1.24.3
scikit-learn==1.3.0
```
---
## DynamoDB Table Design

**Table Name:** `vpbank-entity-risk`
**Schema:**
- **Partition Key:** `entity_key` (String) - Format: `{entity_type}#{entity_id}`
- Example: `id_23#1`, `DeviceInfo#device_abc123`, `P_emaildomain#gmail.com`
**Attributes:**
- `entity_key` (S) - Partition key
- `entity_type` (S) - Type: id_23, DeviceInfo, id_30, id_31, id_33, P_emaildomain, card1
- `entity_id` (S) - Entity identifier
- `risk_score` (N) - Combined risk score (0.0-1.0)
- `fraud_rate` (N) - Historical fraud rate
- `anomaly_score` (N) - Autoencoder anomaly score
- `transaction_count` (N) - Number of transactions
**Indexes:** None (simple key-value lookup)
**Capacity:** On-Demand (auto-scaling)
**Sample Items:**
```json
{
"entity_key": "id_23#1",
"entity_type": "id_23",
"entity_id": "1",
"risk_score": 0.4208,
"fraud_rate": 0.1333,
"anomaly_score": 0.5123,
"transaction_count": 180
}
```
---
## Security & Best Practices

1. **API Key Authentication**
- Generate API key in API Gateway
- Require `X-Api-Key` header for all requests
2. **IAM Roles (Least Privilege)**
- Lambda execution role: S3 read, DynamoDB read only
- No write permissions to models or entity risk
3. **Encryption**
- S3: Server-side encryption (SSE-S3)
- DynamoDB: Encryption at rest enabled
- API Gateway: HTTPS only
4. **VPC (Optional)**
- For production, place Lambda in VPC
- Use VPC endpoints for S3/DynamoDB (no internet gateway)
5. **Monitoring**
- CloudWatch alarms for:
- Lambda errors > 1%
- API Gateway 5xx errors
- Lambda duration > 5 seconds (cold start issue)
- DynamoDB throttling
---
## Performance Targets

| Metric | Target | AWS Service Impact |
|----------------|-----------|------------------------------|
| Cold start | <3s | Lambda (first invocation) |
| Warm latency | <100ms | Lambda (cached model) |
| P95 latency | <150ms | Lambda + DynamoDB |
| Throughput | 1000 TPS | API Gateway limit |
| Availability | 99.9% | AWS SLA |
---
**Next:** Implementation scripts and deployment guide!
