# VPBank StreamGuard - AWS Deployment Guide
**Target:** Deploy serverless fraud detection API to AWS
**Budget:** $100 credit (~$1-5/month usage = 20+ months)
**Architecture:** DynamoDB + Lambda + API Gateway + S3
---
## Prerequisites
### 1. AWS Account Setup
- [ ] AWS account created with $100 credit
- [ ] AWS CLI installed
- [ ] AWS credentials configured
```bash
# Install AWS CLI (if not installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
# Verify installation
aws --version
# Configure AWS credentials
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output format (json)
# Test connection
aws sts get-caller-identity
```
### 2. Install AWS SAM CLI
```bash
# Install SAM CLI (Serverless Application Model)
pip install aws-sam-cli
# Verify installation
sam --version
```
### 3. Python Dependencies
```bash
# Install required packages
pip install boto3 pandas
```
---
## Deployment Steps
### Step 1: Prepare Models (5 min)
**Check Fast Lane model exists:**
```bash
# Find latest Fast Lane model
ls -lh models/fast_lane_baseline_*/
# Should see:
# - lr_model.pkl (Logistic Regression model)
# - scaler.pkl (StandardScaler)
```
**Set model directory:**
```bash
# Example (update with your actual directory)
export MODEL_DIR="models/fast_lane_baseline_20251105_210615"
# Verify files exist
ls -lh $MODEL_DIR/lr_model.pkl
ls -lh $MODEL_DIR/scaler.pkl
```
---
### Step 2: Deploy Infrastructure with SAM (10 min)
**Navigate to AWS directory:**
```bash
cd aws/
```
**Build SAM application:**
```bash
# Build Lambda function package
sam build
# This will:
# - Create .aws-sam/build/ directory
# - Install Python dependencies from lambda/requirements.txt
# - Package Lambda function code
```
**Deploy to AWS:**
```bash
# Deploy with guided mode (first time)
sam deploy --guided
# You will be prompted for:
# - Stack Name: vpbank-fraud-detection
# - AWS Region: us-east-1 (or your preferred region)
# - Confirm changes before deploy: Y
# - Allow SAM CLI IAM role creation: Y
# - Disable rollback: N
# - Save arguments to configuration file: Y
# - SAM configuration file: samconfig.toml
# - SAM configuration environment: default
# Deployment will take ~3-5 minutes
```
**Wait for deployment to complete:**
```
Deploying with following values
===============================
Stack name : vpbank-fraud-detection
Region : us-east-1
Confirm changeset : True
Deployment s3 bucket : aws-sam-cli-managed-default-samclisourcebucket-xxx
Capabilities : ["CAPABILITY_IAM"]
Parameter overrides : {}
Initiating deployment
=====================
CloudFormation stack changeset
-------------------------------------------------------------------------------------------------
Operation LogicalResourceId ResourceType
-------------------------------------------------------------------------------------------------
+ Add EntityRiskTable AWS::DynamoDB::Table
+ Add ModelBucket AWS::S3::Bucket
+ Add FraudScoringFunctionRole AWS::IAM::Role
+ Add FraudScoringFunction AWS::Lambda::Function
+ Add FraudAPI AWS::ApiGateway::RestApi
-------------------------------------------------------------------------------------------------
Changeset created successfully. arn:aws:cloudformation:us-east-1:xxx:changeSet/xxx
Deploy this changeset? [y/N]: y
CloudFormation events from stack operations
-------------------------------------------------------------------------------------------------
ResourceStatus ResourceType LogicalResourceId
-------------------------------------------------------------------------------------------------
CREATE_IN_PROGRESS AWS::DynamoDB::Table EntityRiskTable
CREATE_IN_PROGRESS AWS::S3::Bucket ModelBucket
CREATE_IN_PROGRESS AWS::IAM::Role FraudScoringFunctionRole
...
CREATE_COMPLETE AWS::CloudFormation::Stack vpbank-fraud-detection
-------------------------------------------------------------------------------------------------
Successfully created/updated stack - vpbank-fraud-detection in us-east-1
```
**Capture outputs:**
```bash
# Get stack outputs
aws cloudformation describe-stacks \
--stack-name vpbank-fraud-detection \
--query 'Stacks[0].Outputs' \
--output table
# Save outputs to file
aws cloudformation describe-stacks \
--stack-name vpbank-fraud-detection \
--query 'Stacks[0].Outputs' > outputs.json
# Example outputs:
# - ApiUrl: https://xxxxx.execute-api.us-east-1.amazonaws.com/prod
# - PredictEndpoint: https://xxxxx.execute-api.us-east-1.amazonaws.com/prod/predict
# - BucketName: vpbank-fraud-models-123456789012
# - TableName: vpbank-entity-risk
```
**Set environment variables:**
```bash
# Extract values from outputs
export API_URL=$(aws cloudformation describe-stacks --stack-name vpbank-fraud-detection --query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text)
export BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name vpbank-fraud-detection --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' --output text)
export TABLE_NAME=$(aws cloudformation describe-stacks --stack-name vpbank-fraud-detection --query 'Stacks[0].Outputs[?OutputKey==`TableName`].OutputValue' --output text)
# Verify
echo "API URL: $API_URL"
echo "Bucket: $BUCKET_NAME"
echo "Table: $TABLE_NAME"
```
---
### Step 3: Upload Models to S3 (2 min)
**Run upload script:**
```bash
cd ../ # Back to project root
python aws/scripts/upload_models_to_s3.py \
--bucket $BUCKET_NAME \
--model-dir $MODEL_DIR
```
**Expected output:**
```
================================================================================
Uploading Models to S3
================================================================================
Bucket: vpbank-fraud-models-123456789012
Model directory: models/fast_lane_baseline_20251105_210615
Uploading models/fast_lane_baseline_20251105_210615/lr_model.pkl (0.02 MB) to s3://vpbank-fraud-models-123456789012/fast_lane/lr_model.pkl...
Uploaded successfully
Uploading models/fast_lane_baseline_20251105_210615/scaler.pkl (0.01 MB) to s3://vpbank-fraud-models-123456789012/fast_lane/scaler.pkl...
Uploaded successfully
================================================================================
Verifying Uploads
================================================================================
s3://vpbank-fraud-models-123456789012/fast_lane/lr_model.pkl (0.02 MB)
s3://vpbank-fraud-models-123456789012/fast_lane/scaler.pkl (0.01 MB)
================================================================================
Upload Complete!
================================================================================
```
---
### Step 4: Load Entity Risk to DynamoDB (3 min)
**Find entity risk CSV:**
```bash
# Find latest Deep Lane model directory
export DEEP_LANE_DIR="models/deep_lane_20251109_020030"
# Verify CSV exists
ls -lh $DEEP_LANE_DIR/entity_risk_combined.csv
# Should show ~5,526 rows
wc -l $DEEP_LANE_DIR/entity_risk_combined.csv
```
**Run load script:**
```bash
python aws/scripts/load_entity_risk_to_dynamodb.py \
--table $TABLE_NAME \
--csv $DEEP_LANE_DIR/entity_risk_combined.csv
```
**Expected output:**
```
================================================================================
Loading Entity Risk to DynamoDB
================================================================================
Table: vpbank-entity-risk
CSV: models/deep_lane_20251109_020030/entity_risk_combined.csv
Reading CSV...
Loaded 5,526 entities from CSV
Sample data:
entity_type entity_id risk_score fraud_rate anomaly_score transaction_count
id_23 1 0.420811 0.034685 0.512345 99121
DeviceInfo 0 0.000000 0.000000 0.000000 1234
Preparing items for DynamoDB...
Prepared 5,526 items
Writing to DynamoDB in batches of 25...
Batch 1/221: Wrote 25 items
Batch 2/221: Wrote 25 items
...
Batch 221/221: Wrote 6 items
================================================================================
Load Complete!
================================================================================
Successful: 5,526
Failed: 0
Total: 5,526
Verifying data...
Table contains 5,526 items
Testing sample queries:
id_23#1: risk_score=0.4208
DeviceInfo#0: risk_score=0.0000
P_emaildomain#0: risk_score=0.4186
================================================================================
Entity risk data loaded successfully!
================================================================================
```
---
### Step 5: Test API (5 min)
**Create test script:**
```bash
# Create test file
cat > test_api.sh << 'EOF'
#!/bin/bash
# Test API endpoint
API_URL=$1
echo "Testing VPBank StreamGuard API"
echo "API URL: $API_URL"
echo ""
# Test 1: Health Check
echo "Test 1: Health Check"
curl -s "$API_URL/health" | jq .
echo ""
# Test 2: Prediction with entity risk
echo "Test 2: Prediction with Entity Risk"
curl -s -X POST "$API_URL/predict" \
-H "Content-Type: application/json" \
-d '{
"V1": -1.3598071336738,
"V2": -0.0727811733098497,
"V3": 2.53634673796914,
"V4": 1.37815522427443,
"V5": -0.338320769942518,
"V6": 0.462387777762292,
"V7": 0.239598554061257,
"V8": 0.0986979012610507,
"V9": 0.363786969611213,
"V10": 0.0907941719789316,
"V11": -0.551599533260813,
"V12": -0.617800855762348,
"V13": -0.991389847235408,
"V14": -0.311169353699879,
"V15": 1.46817697209427,
"V16": -0.470400525259478,
"V17": 0.207971241929242,
"V18": 0.0257905801985591,
"V19": 0.403992960255733,
"V20": 0.251412098239705,
"V21": -0.018306777944153,
"V22": 0.277837575558899,
"V23": -0.110473910188767,
"V24": 0.0669280749146731,
"V25": 0.128539358273528,
"V26": -0.189114843888824,
"V27": 0.133558376740387,
"V28": -0.0210530534538215,
"Time": 406,
"Amount": 100.0,
"ip_proxy": "1",
"device_id": "device_123"
}' | jq .
echo ""
# Test 3: Prediction without entity risk
echo "Test 3: Prediction without Entity Risk"
curl -s -X POST "$API_URL/predict" \
-H "Content-Type: application/json" \
-d '{
"V1": -1.3598071336738,
"V2": -0.0727811733098497,
"V3": 2.53634673796914,
"V4": 1.37815522427443,
"V5": -0.338320769942518,
"V6": 0.462387777762292,
"V7": 0.239598554061257,
"V8": 0.0986979012610507,
"V9": 0.363786969611213,
"V10": 0.0907941719789316,
"V11": -0.551599533260813,
"V12": -0.617800855762348,
"V13": -0.991389847235408,
"V14": -0.311169353699879,
"V15": 1.46817697209427,
"V16": -0.470400525259478,
"V17": 0.207971241929242,
"V18": 0.0257905801985591,
"V19": 0.403992960255733,
"V20": 0.251412098239705,
"V21": -0.018306777944153,
"V22": 0.277837575558899,
"V23": -0.110473910188767,
"V24": 0.0669280749146731,
"V25": 0.128539358273528,
"V26": -0.189114843888824,
"V27": 0.133558376740387,
"V28": -0.0210530534538215,
"Time": 406,
"Amount": 100.0
}' | jq .
echo ""
echo "API testing complete!"
EOF
chmod +x test_api.sh
```
**Run tests:**
```bash
./test_api.sh $API_URL
```
**Expected output:**
```
Testing VPBank StreamGuard API
API URL: https://xxxxx.execute-api.us-east-1.amazonaws.com/prod
Test 1: Health Check
{
"status": "healthy",
"model_loaded": true,
"dynamodb_connected": true,
"timestamp": "2025-11-09T12:34:56Z"
}
Test 2: Prediction with Entity Risk
{
"risk_score": 0.4521,
"model_score": 0.3245,
"entity_risk": 0.4208,
"entity_risks": {
"ip_proxy": 0.4208,
"device": 0.0
},
"decision": "challenge",
"timestamp": "2025-11-09T12:34:56Z"
}
Test 3: Prediction without Entity Risk
{
"risk_score": 0.2271,
"model_score": 0.3245,
"entity_risk": 0.0,
"entity_risks": {},
"decision": "pass",
"timestamp": "2025-11-09T12:34:56Z"
}
API testing complete!
```
---
## Deployment Verification
### Check All Resources
**1. DynamoDB Table:**
```bash
aws dynamodb describe-table --table-name $TABLE_NAME --query 'Table.[TableName,ItemCount,TableSizeBytes]'
# Expected: ["vpbank-entity-risk", 5526, ~500000]
```
**2. S3 Bucket:**
```bash
aws s3 ls s3://$BUCKET_NAME/fast_lane/
# Expected:
# lr_model.pkl
# scaler.pkl
```
**3. Lambda Function:**
```bash
aws lambda get-function --function-name vpbank-fraud-scoring --query 'Configuration.[FunctionName,Runtime,MemorySize,Timeout]'
# Expected: ["vpbank-fraud-scoring", "python3.11", 512, 10]
```
**4. API Gateway:**
```bash
aws apigateway get-rest-apis --query 'items[?name==`vpbank-fraud-api`].[name,id]'
# Expected: [["vpbank-fraud-api", "xxxxx"]]
```
---
## Cost Monitoring
### Set up Billing Alarm
```bash
# Create SNS topic for billing alerts
aws sns create-topic --name billing-alerts
# Subscribe to topic
aws sns subscribe \
--topic-arn arn:aws:sns:us-east-1:123456789012:billing-alerts \
--protocol email \
--notification-endpoint your-email@example.com
# Create billing alarm (alert if cost > $10/month)
aws cloudwatch put-metric-alarm \
--alarm-name billing-alert-10-dollars \
--alarm-description "Alert when monthly bill exceeds $10" \
--metric-name EstimatedCharges \
--namespace AWS/Billing \
--statistic Maximum \
--period 21600 \
--evaluation-periods 1 \
--threshold 10 \
--comparison-operator GreaterThanThreshold \
--alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts
```
### Check Current Costs
```bash
# View cost explorer (requires Cost Explorer enabled in AWS Console)
aws ce get-cost-and-usage \
--time-period Start=2025-11-01,End=2025-11-30 \
--granularity MONTHLY \
--metrics BlendedCost \
--group-by Type=SERVICE
```
---
## Troubleshooting
### Lambda Function Logs
```bash
# View recent logs
aws logs tail /aws/lambda/vpbank-fraud-scoring --follow
# View specific log stream
aws logs get-log-events \
--log-group-name /aws/lambda/vpbank-fraud-scoring \
--log-stream-name 2025/11/09/[$LATEST]xxxxx
```
### API Gateway Logs
```bash
# Enable logging (if not already enabled)
aws logs describe-log-groups --log-group-name-prefix /aws/apigateway/
```
### Common Issues
**1. Lambda function can't load model:**
- Check S3 bucket permissions
- Verify model files uploaded correctly
- Check Lambda execution role has S3 read permission
**2. DynamoDB query returns nothing:**
- Verify table has items: `aws dynamodb scan --table-name $TABLE_NAME --select COUNT`
- Check entity_key format: Should be `{type}#{id}` (e.g., `id_23#1`)
**3. API Gateway 403 Forbidden:**
- Check if API key is required (disable for testing)
- Verify CORS settings if calling from browser
---
## Next Steps
1. **Monitor costs** - Check AWS Billing Dashboard daily for first week
2. **Add API Key authentication** - Update SAM template to require API keys
3. **Enable CloudWatch alarms** - Alert on Lambda errors, high latency
4. **Load test** - Use `wrk` or `locust` to test throughput
5. **Optimize** - Tune Lambda memory, add caching if needed
---
## Cleanup (Delete Stack)
**To delete all resources and stop incurring costs:**
```bash
# Delete CloudFormation stack
aws cloudformation delete-stack --stack-name vpbank-fraud-detection
# Wait for deletion
aws cloudformation wait stack-delete-complete --stack-name vpbank-fraud-detection
# Manually delete S3 bucket (must be empty first)
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME
# Verify all resources deleted
aws cloudformation describe-stacks --stack-name vpbank-fraud-detection
# Should return: Stack does not exist
```
---
## Deployment Summary
| Resource | Status | Details |
|------------------|--------|--------------------------------------|
| DynamoDB Table | | 5,526 items, PAY_PER_REQUEST |
| S3 Bucket | | 2 model files (~30 KB) |
| Lambda Function | | Python 3.11, 512 MB, 10s timeout |
| API Gateway | | REST API, /predict + /health |
| IAM Roles | | Least privilege permissions |
| CloudWatch Logs | | 7-day retention |
**Total Cost:** ~$0.11/month (within free tier for first 12 months)
**Budget Remaining:** $99.89 (~909 months at current usage)
---
**Deployment Complete! **
Your fraud detection API is now live at: `$API_URL/predict`
Test it with the provided script or integrate with your application!
