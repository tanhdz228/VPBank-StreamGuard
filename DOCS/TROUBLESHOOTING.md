# VPBank StreamGuard - Troubleshooting Guide
## Quick Diagnostics
### Check System Health
```bash
# 1. Check Python version
python --version
# Should be: Python 3.8 or higher
# 2. Check installed packages
pip list | grep -E "(streamlit|scikit-learn|boto3|pandas)"
# 3. Test API (if deployed)
curl https://your-api-url.amazonaws.com/prod/health
```
## Common Issues
### 1. Demo Won't Start
#### Symptom
```
'python' is not recognized as an internal or external command
```
#### Cause
Python not installed or not in PATH
#### Solution
**Windows:**
1. Download Python from https://www.python.org/downloads/
2. Run installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Restart terminal
**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip
# Mac
brew install python3
```
---
#### Symptom
```
ERROR: Could not find a version that satisfies the requirement streamlit
```
#### Cause
Outdated pip or network issue
#### Solution
```bash
# Upgrade pip
python -m pip install --upgrade pip
# Install with verbose output
pip install streamlit --verbose
```
---
#### Symptom
```
ModuleNotFoundError: No module named 'streamlit'
```
#### Cause
Packages not installed or wrong Python environment
#### Solution
```bash
# Verify pip is installing to correct Python
which python
which pip
# Install packages explicitly
pip install streamlit numpy pandas scikit-learn boto3 plotly
# Or install from requirements
cd SOURCE
pip install -r requirements.txt
```
### 2. Demo Runs But Shows Errors
#### Symptom
```
FileNotFoundError: Model file not found
```
#### Cause
Models not in expected location
#### Solution
```bash
# Check model files exist
ls ../SOURCE/src/ # Should see model files
# Or download pre-trained models
cd MODELS
# Download from S3 or Google Drive (link provided separately)
```
---
#### Symptom
```
AttributeError: 'NoneType' object has no attribute 'predict'
```
#### Cause
Model failed to load
#### Solution
1. Check model file integrity:
```bash
python -c "import pickle; model = pickle.load(open('model.pkl', 'rb')); print(model)"
```
2. Retrain model:
```bash
cd SOURCE/scripts
python train_fast_lane.py
```
---
#### Symptom
```
streamlit: command not found
```
#### Cause
Streamlit not in PATH
#### Solution
```bash
# Find Streamlit location
pip show streamlit
# Run with full path
python -m streamlit run streamlit_app.py
```
### 3. AWS Deployment Issues
#### Symptom
```
Error: AWS credentials not found
```
#### Cause
AWS credentials not configured
#### Solution
```bash
# Configure AWS CLI
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)
```
---
#### Symptom
```
Error: sam: command not found
```
#### Cause
AWS SAM CLI not installed
#### Solution
```bash
# Install SAM CLI
pip install aws-sam-cli
# Verify
sam --version
```
---
#### Symptom
```
Error: Stack creation failed
```
#### Cause
Various (permissions, resource limits, naming conflicts)
#### Solution
1. Check CloudFormation console for detailed error
2. Common fixes:
```bash
# Delete existing stack
aws cloudformation delete-stack --stack-name vpbank-fraud-detection
# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name vpbank-fraud-detection
# Redeploy
sam deploy --guided
```
---
#### Symptom
```
Lambda function cold start >10s
```
#### Cause
Large deployment package or model loading
#### Solution
1. Optimize model size:
```python
# Use joblib compression
import joblib
joblib.dump(model, 'model.pkl', compress=3)
```
2. Use Lambda layers for dependencies
3. Increase Lambda memory (more CPU = faster cold start)
---
#### Symptom
```
DynamoDB: ProvisionedThroughputExceededException
```
#### Cause
Too many requests to DynamoDB
#### Solution
```yaml
# In template.yaml, change to PAY_PER_REQUEST
BillingMode: PAY_PER_REQUEST
```
### 4. Model Training Issues
#### Symptom
```
MemoryError during XGBoost training
```
#### Cause
Insufficient RAM
#### Solution
1. Reduce dataset size:
```python
# Sample data
df_sample = df.sample(frac=0.5, random_state=42)
```
2. Reduce tree depth:
```python
params = {
'max_depth': 3, # Instead of 6
'colsample_bytree': 0.5 # Use fewer features per tree
}
```
3. Use incremental learning
---
#### Symptom
```
ValueError: Input contains NaN
```
#### Cause
Missing values in data
#### Solution
```python
# Check for NaN
print(df.isnull().sum())
# Fill NaN
df.fillna(df.median(), inplace=True)
# Or use preprocessor
from src.data.creditcard_preprocessor import CreditCardPreprocessor
preprocessor = CreditCardPreprocessor()
X_clean = preprocessor.preprocess(X)
```
---
#### Symptom
```
Autoencoder training very slow
```
#### Cause
No GPU or large dataset
#### Solution
1. Use GPU if available:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
2. Reduce epochs:
```python
model.fit(X_train, X_train, epochs=10) # Instead of 50
```
3. Reduce model size:
```python
encoder = Sequential([
Dense(64, activation='relu'), # Instead of 128
Dense(32, activation='relu') # Instead of 64
])
```
### 5. API / Performance Issues
#### Symptom
```
Latency >500ms
```
#### Cause
Cold start or slow model
#### Solution
1. Keep Lambda warm:
```bash
# Use CloudWatch Event to ping every 5 minutes
aws events put-rule --name keep-lambda-warm --schedule-rate "rate(5 minutes)"
```
2. Optimize model:
```python
# Use lighter model (Logistic instead of XGBoost)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```
3. Cache entity risk:
```python
# In Lambda, cache DynamoDB results
entity_risk_cache = {}
```
---
#### Symptom
```
API returns 502 Bad Gateway
```
#### Cause
Lambda timeout or error
#### Solution
1. Check CloudWatch logs:
```bash
aws logs tail /aws/lambda/vpbank-fraud-scoring --follow
```
2. Increase timeout:
```yaml
# In template.yaml
Timeout: 30 # Instead of 10
```
3. Fix Lambda code errors
---
#### Symptom
```
High AWS costs
```
#### Cause
Excessive API calls or large Lambda memory
#### Solution
1. Check CloudWatch metrics:
```bash
aws cloudwatch get-metric-statistics \
--namespace AWS/Lambda \
--metric-name Invocations \
--start-time 2025-11-01T00:00:00Z \
--end-time 2025-11-10T23:59:59Z \
--period 86400 \
--statistics Sum
```
2. Optimize Lambda memory:
```yaml
# Start with 512 MB, not 1024 MB
MemorySize: 512
```
3. Set up billing alerts:
```bash
aws cloudwatch put-metric-alarm \
--alarm-name high-cost \
--comparison-operator GreaterThanThreshold \
--threshold 50 \
--evaluation-periods 1 \
--metric-name EstimatedCharges
```
### 6. Data Issues
#### Symptom
```
AUC very low (<0.6)
```
#### Cause
Data leakage, bad features, or class imbalance
#### Solution
1. Check class distribution:
```python
print(y.value_counts())
# Should be ~0.17% fraud for Credit Card, ~3.5% for IEEE-CIS
```
2. Check feature distribution:
```python
df.describe()
```
3. Verify no data leakage:
```python
# Don't include future information or target-derived features
```
---
#### Symptom
```
PSI > 0.25 (data drift detected)
```
#### Cause
Model trained on old data or synthetic mismatch
#### Solution
1. Retrain model:
```bash
cd SOURCE/scripts
python train_fast_lane.py
python train_deep_lane.py
```
2. Update thresholds:
```bash
python run_optimization.py
```
3. Monitor production data distribution
### 7. Dashboard Issues
#### Symptom
Dashboard shows "Connection Error"
#### Cause
API URL incorrect or API down
#### Solution
1. Check API URL:
```python
# In streamlit_app.py
API_URL = "https://correct-url.amazonaws.com/prod"
```
2. Test API manually:
```bash
curl https://your-api-url.amazonaws.com/prod/health
```
3. Use local fallback:
```python
# Dashboard will automatically use local scoring if API fails
```
---
#### Symptom
"Details" button doesn't work
#### Cause
State management issue in Streamlit
#### Solution
Already fixed in current version. If issue persists:
```python
# Add st.rerun() after button click
if st.button("View Details"):
st.session_state.show_details = True
st.rerun()
```
## Debug Tools
### 1. Test API Locally
```python
# test_api.py
import requests
import json
API_URL = "http://localhost:8501/predict" # Or AWS URL
transaction = {
"V1": -1.36, "V2": -0.07, # ... all V features
"Time": 406,
"Amount": 100.0
}
response = requests.post(API_URL, json=transaction)
print(json.dumps(response.json(), indent=2))
```
### 2. Test Model Locally
```python
# test_model.py
import pickle
import numpy as np
# Load model
with open('lr_model.pkl', 'rb') as f:
model = pickle.load(f)
# Test prediction
X_test = np.random.randn(1, 30) # 30 features
score = model.predict_proba(X_test)[0, 1]
print(f"Risk score: {score:.4f}")
```
### 3. Check Entity Risk
```python
# test_entity_risk.py
import boto3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('vpbank-entity-risk')
# Get entity
response = table.get_item(Key={'entity_key': 'id_23#1'})
print(response.get('Item', 'Not found'))
```
### 4. Monitor Lambda
```bash
# Tail Lambda logs
aws logs tail /aws/lambda/vpbank-fraud-scoring --follow
# Get metrics
aws cloudwatch get-metric-statistics \
--namespace AWS/Lambda \
--metric-name Duration \
--dimensions Name=FunctionName,Value=vpbank-fraud-scoring \
--start-time 2025-11-10T00:00:00Z \
--end-time 2025-11-10T23:59:59Z \
--period 3600 \
--statistics Average,Maximum
```
## Still Having Issues?
### Collect Diagnostic Info
```bash
# System info
python --version
pip list
# Package versions
pip show streamlit scikit-learn boto3
# AWS config
aws configure list
# Model files
ls -lh models/
# Recent logs
tail -n 50 demo/logs/streamlit.log
```
### Contact Support
When reporting issues, include:
1. Operating system & Python version
2. Full error message and stack trace
3. Steps to reproduce
4. Relevant log files
5. Command output from "Collect Diagnostic Info" above
## Useful Commands
```bash
# Reset demo
rm -rf .streamlit/
rm -rf __pycache__/
# Reinstall packages
pip uninstall -y streamlit
pip install streamlit
# Clear pip cache
pip cache purge
# Test Python environment
python -c "import sys; print(sys.executable)"
# Check AWS credentials
aws sts get-caller-identity
# Delete all AWS resources
aws cloudformation delete-stack --stack-name vpbank-fraud-detection
```
## Performance Benchmarks
Expected performance on modern laptop:
| Task | Time | Notes |
|------|------|-------|
| Demo startup | <10s | After first run |
| Transaction scoring | <100ms | Local |
| Model training (Fast) | 2-5 min | Credit Card dataset |
| Model training (Deep) | 30-60 min | IEEE-CIS dataset |
| AWS deployment | 10-15 min | SAM deploy |
If significantly slower, check:
- CPU/RAM usage
- Disk space
- Antivirus interference
- Network latency (for AWS)
