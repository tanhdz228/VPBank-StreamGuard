# VPBank StreamGuard - Interactive Demo
## Quick Start (60 seconds)
### Windows Users

1. Double-click `run_demo.bat`
2. Wait for the browser to open (or go to http://localhost:8501)
3. Start testing fraud detection!
### Linux/Mac Users

1. Open terminal in this folder
2. Run: `./run_demo.sh`
3. Wait for the browser to open (or go to http://localhost:8501)
4. Start testing fraud detection!
## What This Demo Does

This interactive dashboard demonstrates the **VPBank StreamGuard** fraud detection system:
- **Real-time Fraud Scoring**: Test transactions and get instant risk scores
- **Dual-Track Architecture**: See both Fast Lane and Deep Lane models in action
- **Explainable AI**: Understand why transactions are flagged as risky
- **Risk-Based Authentication**: Automatic decision (Pass/Challenge/Block)
- **Analytics Dashboard**: View trends and statistics
## Demo Features
### 1. Manual Transaction Testing

- Enter transaction details (amount, time, channel, etc.)
- Get instant fraud risk score (0-100)
- See decision: Green (Pass), Yellow (Challenge), Red (Block)
- View top 5 risk factors with SHAP explanations
### 2. Automated Transaction Stream

- Simulate real-time transaction flow
- Watch fraud detection in action
- See distribution of risk scores
- Monitor Pass/Challenge/Block decisions
### 3. Test Scenarios

Try these pre-configured scenarios:
**Green (Low Risk - Pass)**
- Amount: $50
- Time: 2 PM
- Channel: Mobile app
- Device: Known device
- Expected: Risk ~10-20%, Pass
**Yellow (Medium Risk - Challenge)**
- Amount: $500
- Time: 9 PM
- Channel: Web
- Device: New device
- Expected: Risk ~40-60%, Challenge (OTP required)
**Red (High Risk - Block)**
- Amount: $2,000
- Time: 2 AM
- Channel: Web
- Device: Fraud-linked device
- IP: Proxy/VPN
- Expected: Risk ~80-90%, Block
## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2 GB minimum
- **Internet**: Required for package installation (first run only)
## First Run Notes

The first time you run the demo:
1. It will install required Python packages (~5 minutes)
2. It will download pre-trained models if needed
3. Subsequent runs will be much faster (<10 seconds)
## Troubleshooting

**"Python is not installed"**
- Install Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation
**"pip install failed"**
- Try running as administrator (Windows) or with sudo (Linux/Mac)
- Or manually run: `pip install streamlit numpy pandas scikit-learn boto3 plotly`
**"Port 8501 is already in use"**
- Another Streamlit app is running
- Stop the other app or use: `streamlit run streamlit_app.py --server.port 8502`
**Demo won't open in browser**
- Manually navigate to http://localhost:8501
- Check firewall settings
## What's Happening Behind the Scenes?

When you score a transaction, the demo:
1. **Feature Engineering** (10ms)
- Converts raw inputs to 34 ML features
- Scales and normalizes values
2. **Fast Lane Scoring** (20ms)
- Logistic Regression model trained on Credit Card dataset
- Produces transaction-level risk score
3. **Deep Lane Lookup** (10ms)
- Fetches entity risk from pre-computed database
- Uses device, IP, and email domain reputation
4. **Score Combination** (1ms)
- Combines: 70% Fast Lane + 30% Deep Lane
- Produces final risk score (0-1)
5. **RBA Policy Decision** (1ms)
- < 30%: **Pass** (green)
- 30-70%: **Challenge** (yellow, step-up auth)
- > 70%: **Block** (red, manual review)
6. **SHAP Explanation** (50ms)
- Calculates feature contributions
- Generates human-readable reason codes
**Total Latency**: ~90ms (meets <150ms SLA)
## Demo vs Production

This demo runs **locally** on your machine and uses:
- Pre-trained models (frozen, not learning)
- Simulated entity risk data
- Mock transaction generator
The **production system** on AWS uses:
- Live API endpoint (50-150ms latency)
- Real-time entity risk updates (every 15-30 min)
- Actual transaction streams from banking channels
- Continuous model monitoring and retraining
## Next Steps

After exploring the demo:
1. **Review Source Code**: Check `../SOURCE/` for implementation
2. **Read Documentation**: See `../DOCS/` for technical details
3. **Deploy to AWS**: Follow `../SOURCE/aws/` deployment guide
4. **Train Your Own Models**: Use scripts in `../SOURCE/scripts/`
## Demo Architecture

```
streamlit_app.py
Load models (Fast Lane: Logistic Regression)
Load entity risk (5,526 pre-computed entities)
Generate mock transaction
Feature engineering (V1-V28 + derived features)
Score with Fast Lane model
Lookup entity risk (device, IP, email)
Combine scores (70% model + 30% entity)
Apply RBA policy (pass/challenge/block)
Generate SHAP explanations
Display results in web UI
```
## Support

For questions or issues:
- Technical: Review `../DOCS/TROUBLESHOOTING.md`
- Business: Review `../DOCS/ROI_ANALYSIS.md`
- Deployment: Review `../SOURCE/aws/AWS_DEPLOYMENT_GUIDE.md`
---
**Enjoy the demo!** 
This is a fully functional fraud detection system running on your local machine.
