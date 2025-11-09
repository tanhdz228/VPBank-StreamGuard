# VPBank StreamGuard - Pre-trained Models
## Overview

This folder contains pre-trained machine learning models and entity risk data for the VPBank StreamGuard fraud detection system.
## Model Structure

```
MODELS/
│
├── README.md                      # This file (model documentation)
│
├── fast_lane/                     # Fast Lane (Real-time Scoring)
│   ├── logistic_model.pkl         # Logistic Regression model (1 KB)
│   └── preprocessor.pkl           # StandardScaler for features (2 KB)
│
└── deep_lane/                     # Deep Lane (Batch Processing)
    └── entity_risk_combined.csv   # 5,526 pre-computed entity risks (452 KB)
```
## Fast Lane Models
### logistic_model.pkl

- **Purpose**: Real-time transaction scoring
- **Type**: Scikit-learn Logistic Regression
- **Input**: 34 features (V1-V28 + Time + Amount + derived features)
- **Output**: Fraud probability (0-1)
- **Performance**:
- **Training AUC**: 0.9926
- **Validation AUC**: 0.9650
- **Test AUC**: 0.9234
- **Recall@1%FPR**: 0.7959
**Inference Latency**: ~20ms
- **File Size**: ~1 KB
- **Usage**:
```python
import pickle
import numpy as np
# Load model
with open('fast_lane/logistic_model.pkl', 'rb') as f:
model = pickle.load(f)
# Load scaler
with open('fast_lane/preprocessor.pkl', 'rb') as f:
scaler = pickle.load(f)
# Prepare input (34 features)
X = np.array([[...]]) # V1-V28, Time, Amount, Hour, Time_Period, Amount_Log, etc.
# Scale features
X_scaled = scaler.transform(X)
# Predict
fraud_prob = model.predict_proba(X_scaled)[0, 1]
print(f"Fraud probability: {fraud_prob:.4f}")
```
---
### preprocessor.pkl

- **Purpose**: Feature scaling/normalization
- **Type**: Scikit-learn StandardScaler
- **Input**: 34 raw features
- **Output**: 34 scaled features (mean=0, std=1)
- **Note**: Always apply this scaler before passing data to the model!
---
## Deep Lane Data
### entity_risk_combined.csv

- **Purpose**: Pre-computed entity reputation scores
- **Format**: CSV with columns:
- `entity_type`: Type of entity (id_23, DeviceInfo, id_30, id_31, id_33, P_emaildomain, card1)
- `entity_id`: Unique identifier for the entity
- `fraud_rate`: Historical fraud rate for this entity (0-1)
- `total_transactions`: Number of transactions seen
- `risk_score`: Normalized risk score (0-1)
- `anomaly_score`: Autoencoder anomaly score
**Entity Types**:
- **id_23**: IP proxy indicator (4 entities, CRITICAL: 13.3% fraud rate)
- **DeviceInfo**: Device fingerprints (597 entities)
- **id_30**: Operating system (73 entities)
- **id_31**: Browser type (105 entities)
- **id_33**: Screen resolution (90 entities)
- **P_emaildomain**: Email domain (60 entities)
- **card1**: Card identifier (4,597 entities)
**Total Entities**: 5,526
- **High-Risk Entities**: 43 (0.78% with fraud_rate > 0.5)
- **Usage**:
```python
import pandas as pd
# Load entity risk
entity_risk = pd.read_csv('deep_lane/entity_risk_combined.csv')
# Lookup risk for a specific entity
device_risk = entity_risk[
(entity_risk['entity_type'] == 'DeviceInfo') &
(entity_risk['entity_id'] == 'device_123')
]['risk_score'].values[0]
print(f"Device risk: {device_risk:.4f}")
```
**DynamoDB Format** (for AWS deployment):
```python
# Entity key format: {entity_type}#{entity_id}
{
"entity_key": "id_23#1",
"fraud_rate": 0.4208,
"risk_score": 0.4208,
"total_transactions": 1234,
"anomaly_score": 0.6543
}
```
## Model Training Details
### Fast Lane

- **Dataset**: Credit Card Fraud (284,807 transactions, 0.172% fraud)
- **Training Date**: 2025-11-05
- **Training Time**: ~3 minutes
- **Algorithm**: Logistic Regression with L2 regularization
- **Hyperparameters**:
- `C=1.0` (inverse regularization strength)
- `max_iter=1000`
- `class_weight='balanced'` (handle class imbalance)
**Features** (34 total):
- V1-V28: PCA components (from original transaction features)
- Time: Seconds since first transaction
- Amount: Transaction amount
- Hour: Hour of day (derived from Time)
- Time_Period: Time category (morning/afternoon/evening/night)
- Amount_Log: Log-transformed amount
- Other derived features
**Cross-validation**: Stratified 5-fold
### Deep Lane

- **Dataset**: IEEE-CIS Fraud (590,540 transactions, 3.5% fraud)
- **Training Date**: 2025-11-09
- **Training Time**: ~45 minutes
- **Algorithms**:
1. **XGBoost** (supervised)
- Mean CV AUC: 0.8940
- 159 features (V95-V137, V279-V321, identity, transaction features)
- 5-fold GroupKFold (by time)
2. **Autoencoder** (unsupervised)
- Architecture: [159 -> 128 -> 64 -> 32 -> 64 -> 128 -> 159]
- Reconstruction error threshold: 45,661.91
**Entity Risk Calculation**:
```python
# For each entity, aggregate fraud statistics
entity_risk = df.groupby(['entity_type', 'entity_id']).agg({
'isFraud': ['mean', 'sum', 'count'], # fraud_rate, fraud_count, total
'TransactionAmt': 'mean',
'autoencoder_error': 'mean'
}).reset_index()
# Normalize to 0-1 risk score
entity_risk['risk_score'] = normalize(entity_risk['fraud_rate'])
```
## File Sizes & Storage

| File | Size | Storage | Download Time (1 Mbps) |
|------|------|---------|------------------------|
| logistic_model.pkl | 1 KB | S3 | <1 second |
| preprocessor.pkl | 2 KB | S3 | <1 second |
| entity_risk_combined.csv | 452 KB | S3 / DynamoDB | <1 second |
| **Total** | **455 KB** | - | **<2 seconds** |
## Model Versioning

Current version: **v1.0** (2025-11-10)
- **Version Naming**: `{model_type}_{dataset}_{YYYYMMDD_HHMMSS}`
Examples:
- `fast_lane_baseline_20251105_210615`
- `deep_lane_20251109_020030`
## Retraining Models
### Fast Lane

```bash
cd ../SOURCE/scripts
python train_fast_lane.py
# Output:
# - models/fast_lane_baseline_{timestamp}/logistic_model.pkl
# - models/fast_lane_baseline_{timestamp}/preprocessor.pkl
```
### Deep Lane

```bash
cd ../SOURCE/scripts
python train_deep_lane.py
# Output:
# - models/deep_lane_{timestamp}/xgboost/
# - models/deep_lane_{timestamp}/autoencoder/
# - models/deep_lane_{timestamp}/entity_risk_combined.csv
```
## Model Deployment
### Local (Demo)

Models are loaded directly from this folder:
```python
# In demo/streamlit_app.py
MODEL_DIR = '../MODELS/fast_lane'
model = pickle.load(open(f'{MODEL_DIR}/logistic_model.pkl', 'rb'))
```
### AWS Lambda

Models are uploaded to S3 and loaded at runtime:
```bash
# Upload to S3
cd ../SOURCE/aws/scripts
python upload_models_to_s3.py \
--bucket vpbank-fraud-models-{account-id} \
--model-dir ../../MODELS/fast_lane
# Lambda caches in /tmp
# - First invocation: Load from S3 (~500ms)
# - Subsequent: Load from /tmp (~50ms)
```
### DynamoDB (Entity Risk)

```bash
# Load entity risk to DynamoDB
cd ../SOURCE/aws/scripts
python load_entity_risk_to_dynamodb.py \
--table vpbank-entity-risk \
--csv ../../MODELS/deep_lane/entity_risk_combined.csv
# Takes ~3 minutes to load 5,526 entities
```
## Model Performance
### Fast Lane

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| AUC | 0.9926 | **0.9650** | 0.9234 |
| Precision@50%Recall | 0.95 | 0.92 | 0.88 |
| Recall@1%FPR | 0.82 | **0.7959** | 0.74 |
| F1 Score | 0.89 | 0.85 | 0.80 |
### Deep Lane

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** |
|--------|--------|--------|--------|--------|--------|----------|
| AUC | 0.8867 | 0.8866 | 0.8892 | 0.9079 | 0.8996 | **0.8940** |
### Combined System (Expected)

| Metric | Baseline | With StreamGuard | Improvement |
|--------|----------|------------------|-------------|
| Fraud catch rate | 75% | 87-90% | **+12-15%** |
| False positive rate | 2.0% | 0.3-0.5% | **-75-85%** |
| Customer friction | 100K/month | 50K/month | **-50%** |
## Model Limitations
### Fast Lane

- **Domain**: Credit card transactions only
- **Features**: Requires V1-V28 PCA transformation
- **Class imbalance**: Trained on 0.172% fraud rate
- **Drift**: Retrain every 3-6 months
### Deep Lane

- **Coverage**: Only 5,526 entities (unseen entities -> risk_score = 0)
- **Update frequency**: Batch process every 15-30 minutes
- **Cold start**: New devices/IPs have no history
## Security & Privacy

- **Model files contain**:
- Statistical parameters (weights, biases)
- NO raw transaction data
- NO PII (names, emails, phone numbers)
**Entity risk data contains**:
- Aggregated statistics (fraud rates)
- Hashed/tokenized identifiers (device_id, card1)
- NO PII
## Troubleshooting
### "Model failed to load"

```python
# Check model integrity
import pickle
try:
model = pickle.load(open('logistic_model.pkl', 'rb'))
print("Model loaded successfully")
except Exception as e:
print(f"Error: {e}")
```
### "Feature dimension mismatch"

```python
# Check input shape
print(f"Model expects: {model.coef_.shape[1]} features")
print(f"You provided: {X.shape[1]} features")
# Should be 34 features for Fast Lane
```
### "Entity not found"

```python
# Handle missing entities
entity_risk = entity_risk_df[
(entity_risk_df['entity_type'] == 'DeviceInfo') &
(entity_risk_df['entity_id'] == device_id)
]['risk_score'].values
if len(entity_risk) == 0:
entity_risk = 0.0 # Default for unseen entities
else:
entity_risk = entity_risk[0]
```
## Next Steps

1. **Use the models**: See `../DEMO/streamlit_app.py` for integration example
2. **Deploy to AWS**: Follow `../DOCS/DEPLOYMENT_GUIDE.md`
3. **Retrain models**: Use scripts in `../SOURCE/scripts/`
4. **Monitor performance**: Track AUC, precision, recall in production
## Support

For model-related questions:
- Review training code: `../SOURCE/scripts/train_*.py`
- Check model metrics: Training logs and validation results
- See troubleshooting: `../DOCS/TROUBLESHOOTING.md`
