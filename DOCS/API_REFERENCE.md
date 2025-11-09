# VPBank StreamGuard - API Reference
## Overview

The VPBank StreamGuard fraud detection API provides real-time transaction scoring with explainable AI. This document covers all API endpoints, request/response formats, and integration examples.
## Base URL

- **Production**: `https://{api-id}.execute-api.{region}.amazonaws.com/prod`
- **Demo**: `http://localhost:8501` (Streamlit demo)
## Authentication

- **Current**: None (prototype)
- **Production**: API Key in header `X-API-Key: your-api-key`
## Endpoints
### 1. POST /predict

Score a transaction for fraud risk.
#### Request

```http
POST /predict HTTP/1.1
Host: {api-id}.execute-api.{region}.amazonaws.com
Content-Type: application/json
{
"V1": -1.3598,
"V2": -0.0727,
"V3": 2.5363,
...
"V28": -0.0210,
"Time": 406,
"Amount": 100.0,
"ip_proxy": "1", // Optional
"device_id": "device_123", // Optional
"email_domain": "gmail.com" // Optional
}
```
#### Request Fields

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `V1` - `V28` | float | Yes | PCA components from transaction features | -1.3598 |
| `Time` | int | Yes | Seconds elapsed since first transaction | 406 |
| `Amount` | float | Yes | Transaction amount | 100.0 |
| `ip_proxy` | string | No | IP proxy indicator (0 or 1) | "1" |
| `device_id` | string | No | Device identifier | "device_123" |
| `email_domain` | string | No | Email domain | "gmail.com" |
- **Note**: V1-V28 are PCA components. In production, your system would perform PCA transformation before calling the API. For demo/testing, you can use random values between -5 and 5.
#### Response (Success)

```json
{
"risk_score": 0.4521,
"model_score": 0.3245,
"entity_risk": 0.4208,
"entity_risks": {
"ip_proxy": 0.4208,
"device": 0.0,
"email": 0.0
},
"decision": "challenge",
"reason_codes": [
"amount_high",
"entity_risk_high",
"time_unusual",
"device_new",
"ip_proxy"
],
"timestamp": "2025-11-09T12:34:56Z",
"latency_ms": 87
}
```
#### Response Fields

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `risk_score` | float | Final combined risk score | 0.0 - 1.0 |
| `model_score` | float | Fast Lane model score | 0.0 - 1.0 |
| `entity_risk` | float | Maximum entity risk score | 0.0 - 1.0 |
| `entity_risks` | object | Individual entity risk scores | - |
| `decision` | string | RBA policy decision | pass, challenge, block |
| `reason_codes` | array | Top 5 risk factors | - |
| `timestamp` | string | Scoring timestamp (ISO 8601) | - |
| `latency_ms` | int | Processing time in milliseconds | - |
#### Response (Error)

```json
{
"error": "Invalid request format",
"message": "Missing required field: Amount",
"status_code": 400
}
```
#### RBA Decision Logic

| Risk Score | Decision | Action | User Impact |
|------------|----------|--------|-------------|
| < 0.3 | **pass** | Allow transaction immediately | None (seamless) |
| 0.3 - 0.7 | **challenge** | Step-up authentication (OTP/biometric) | 10-15 sec delay |
| > 0.7 | **block** | Block + manual review | Transaction declined |
#### Example Requests

**Low Risk Transaction**
```bash
curl -X POST https://your-api.amazonaws.com/prod/predict \
-H "Content-Type: application/json" \
-d '{
"V1": -0.5, "V2": 0.3, "V3": 1.2, "V4": -0.8,
"V5": 0.6, "V6": -0.4, "V7": 0.9, "V8": 0.2,
"V9": -0.7, "V10": 0.5, "V11": -0.3, "V12": 1.1,
"V13": -0.6, "V14": 0.8, "V15": -0.2, "V16": 0.4,
"V17": -0.9, "V18": 0.7, "V19": -0.1, "V20": 0.3,
"V21": -0.5, "V22": 0.6, "V23": -0.4, "V24": 0.2,
"V25": -0.8, "V26": 0.5, "V27": -0.3, "V28": 0.1,
"Time": 43200,
"Amount": 50.0,
"device_id": "known_device_123"
}'
```
Response:
```json
{
"risk_score": 0.12,
"decision": "pass",
"reason_codes": ["amount_normal", "time_normal", "device_known"]
}
```
**High Risk Transaction**
```bash
curl -X POST https://your-api.amazonaws.com/prod/predict \
-H "Content-Type: application/json" \
-d '{
"V1": 2.5, "V2": -3.2, "V3": 4.1, "V4": -2.8,
"V5": 3.6, "V6": -4.4, "V7": 2.9, "V8": -3.2,
"V9": 4.7, "V10": -2.5, "V11": 3.3, "V12": -4.1,
"V13": 2.6, "V14": -3.8, "V15": 4.2, "V16": -2.4,
"V17": 3.9, "V18": -4.7, "V19": 2.1, "V20": -3.3,
"V21": 4.5, "V22": -2.6, "V23": 3.4, "V24": -4.2,
"V25": 2.8, "V26": -3.5, "V27": 4.3, "V28": -2.1,
"Time": 7200,
"Amount": 2000.0,
"ip_proxy": "1",
"device_id": "fraud_device_789"
}'
```
Response:
```json
{
"risk_score": 0.89,
"decision": "block",
"reason_codes": ["amount_high", "ip_proxy", "time_unusual", "device_fraud", "pattern_anomaly"]
}
```
### 2. GET /health

Check API health status.
#### Request

```http
GET /health HTTP/1.1
Host: {api-id}.execute-api.{region}.amazonaws.com
```
#### Response

```json
{
"status": "healthy",
"version": "1.0",
"uptime_seconds": 3600,
"model_loaded": true,
"entity_risk_count": 5526
}
```
## Reason Codes

The API returns human-readable reason codes explaining why a transaction is risky.
### Amount-Related

| Code | Description | Threshold |
|------|-------------|-----------|
| `amount_high` | Amount significantly above user average | >2 std dev |
| `amount_very_high` | Amount extremely high | >3 std dev |
| `amount_unusual` | Amount pattern is unusual | Statistical anomaly |
### Time-Related

| Code | Description | Threshold |
|------|-------------|-----------|
| `time_unusual` | Transaction at unusual hour | 11 PM - 6 AM |
| `time_pattern_anomaly` | Time pattern breaks user habit | Statistical anomaly |
### Entity-Related (Device, IP, Email)

| Code | Description | Threshold |
|------|-------------|-----------|
| `device_new` | Device never seen before | First transaction |
| `device_fraud` | Device linked to fraud | Fraud rate >10% |
| `ip_proxy` | IP is proxy/VPN/Tor | Proxy detected |
| `ip_fraud` | IP linked to fraud | Fraud rate >10% |
| `email_fraud` | Email domain linked to fraud | Fraud rate >5% |
### Behavioral

| Code | Description | Threshold |
|------|-------------|-----------|
| `pattern_anomaly` | Transaction pattern is anomalous | Autoencoder error >95th percentile |
| `velocity_high` | Too many transactions in short time | >10 tx in 1 hour |
| `sequence_anomaly` | Transaction sequence is unusual | Statistical anomaly |
### Entity Risk

| Code | Description | Threshold |
|------|-------------|-----------|
| `entity_risk_high` | One or more entities have high risk | Entity risk >0.5 |
| `entity_risk_very_high` | One or more entities have very high risk | Entity risk >0.7 |
## Integration Examples
### Python

```python
import requests
import json
API_URL = "https://your-api.amazonaws.com/prod/predict"
def score_transaction(transaction_data):
"""
Score a transaction for fraud risk.
Args:
transaction_data: Dict with V1-V28, Time, Amount, and optional entity IDs
Returns:
Dict with risk_score, decision, and reason_codes
"""
response = requests.post(
API_URL,
headers={"Content-Type": "application/json"},
json=transaction_data,
timeout=5
)
if response.status_code == 200:
return response.json()
else:
raise Exception(f"API error: {response.status_code} - {response.text}")
# Example usage
transaction = {
"V1": -1.36, "V2": -0.07, ..., "V28": -0.02,
"Time": 406,
"Amount": 100.0,
"device_id": "device_123"
}
result = score_transaction(transaction)
print(f"Risk Score: {result['risk_score']:.2%}")
print(f"Decision: {result['decision']}")
print(f"Reasons: {', '.join(result['reason_codes'][:3])}")
# Apply RBA policy
if result['decision'] == 'pass':
allow_transaction()
elif result['decision'] == 'challenge':
request_otp()
else: # block
block_transaction_and_alert()
```
### JavaScript (Node.js)

```javascript
const axios = require('axios');
const API_URL = 'https://your-api.amazonaws.com/prod/predict';
async function scoreTransaction(transactionData) {
try {
const response = await axios.post(API_URL, transactionData, {
headers: { 'Content-Type': 'application/json' },
timeout: 5000
});
return response.data;
} catch (error) {
console.error('API Error:', error.message);
throw error;
}
}
// Example usage
const transaction = {
V1: -1.36, V2: -0.07, /* ... */, V28: -0.02,
Time: 406,
Amount: 100.0,
device_id: 'device_123'
};
scoreTransaction(transaction)
.then(result => {
console.log(`Risk Score: ${(result.risk_score * 100).toFixed(2)}%`);
console.log(`Decision: ${result.decision}`);
console.log(`Reasons: ${result.reason_codes.slice(0, 3).join(', ')}`);
});
```
### Java

```java
import java.net.http.*;
import java.net.URI;
import com.google.gson.*;
public class FraudDetectionClient {
private static final String API_URL = "https://your-api.amazonaws.com/prod/predict";
private static final HttpClient client = HttpClient.newHttpClient();
private static final Gson gson = new Gson();
public static JsonObject scoreTransaction(JsonObject transaction) throws Exception {
HttpRequest request = HttpRequest.newBuilder()
.uri(URI.create(API_URL))
.header("Content-Type", "application/json")
.POST(HttpRequest.BodyPublishers.ofString(gson.toJson(transaction)))
.timeout(Duration.ofSeconds(5))
.build();
HttpResponse<String> response = client.send(request,
HttpResponse.BodyHandlers.ofString());
if (response.statusCode() == 200) {
return gson.fromJson(response.body(), JsonObject.class);
} else {
throw new Exception("API error: " + response.statusCode());
}
}
public static void main(String[] args) throws Exception {
JsonObject transaction = new JsonObject();
transaction.addProperty("V1", -1.36);
// ... add other fields
transaction.addProperty("Time", 406);
transaction.addProperty("Amount", 100.0);
JsonObject result = scoreTransaction(transaction);
System.out.println("Risk Score: " + result.get("risk_score").getAsFloat());
System.out.println("Decision: " + result.get("decision").getAsString());
}
}
```
## Performance & SLA

| Metric | Target | Typical | P95 | P99 |
|--------|--------|---------|-----|-----|
| **Latency** | <150ms | 80ms | 120ms | 150ms |
| **Throughput** | 1000+ TPS | - | - | - |
| **Availability** | 99.9% | 99.95% | - | - |
| **Error Rate** | <0.1% | <0.05% | - | - |
## Rate Limits

- **Current**: 1000 requests/second (burst: 2000)
- **Production**: Configurable via API Gateway
## Error Codes

| Status | Code | Message | Action |
|--------|------|---------|--------|
| 200 | - | Success | - |
| 400 | INVALID_REQUEST | Missing or invalid fields | Check request format |
| 401 | UNAUTHORIZED | Invalid API key | Check authentication |
| 429 | RATE_LIMIT_EXCEEDED | Too many requests | Implement backoff |
| 500 | INTERNAL_ERROR | Server error | Retry with backoff |
| 503 | SERVICE_UNAVAILABLE | Service temporarily down | Retry after 60s |
## Best Practices
### 1. Error Handling

```python
import time
def score_with_retry(transaction, max_retries=3):
for attempt in range(max_retries):
try:
return score_transaction(transaction)
except requests.exceptions.Timeout:
if attempt < max_retries - 1:
time.sleep(2 ** attempt) # Exponential backoff
else:
raise
except requests.exceptions.HTTPError as e:
if e.response.status_code == 429:
time.sleep(60) # Rate limit - wait 1 minute
else:
raise
```
### 2. Batch Processing

For high volume, batch transactions (100-1000 per request):
```python
# Future enhancement - not yet implemented
def score_batch(transactions):
response = requests.post(
f"{API_URL}/batch",
json={"transactions": transactions}
)
return response.json()
```
### 3. Fallback Strategy

```python
def score_with_fallback(transaction):
try:
return score_transaction(transaction)
except Exception as e:
logger.error(f"API failed: {e}")
# Fallback to rule-based system
return rule_based_scoring(transaction)
```
### 4. Monitoring

```python
import time
def score_with_monitoring(transaction):
start = time.time()
try:
result = score_transaction(transaction)
latency = (time.time() - start) * 1000
# Log metrics
logger.info(f"Scored transaction - Latency: {latency:.0f}ms, "
f"Risk: {result['risk_score']:.2%}, "
f"Decision: {result['decision']}")
return result
except Exception as e:
logger.error(f"Scoring failed: {e}")
raise
```
## Changelog
### v1.0 (2025-11-10)

- Initial release
- POST /predict endpoint
- GET /health endpoint
- SHAP-based reason codes
- RBA policy (pass/challenge/block)
## Support

For API issues:
- Check health endpoint: `GET /health`
- Review CloudWatch logs
- Contact: See `TROUBLESHOOTING.md`
