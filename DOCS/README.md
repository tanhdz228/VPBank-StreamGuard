# VPBank StreamGuard - Documentation Index
## Documentation Overview
This folder contains all documentation for the VPBank StreamGuard fraud detection system.
## Quick Links
| Document | Purpose | Audience | Time to Read |
|----------|---------|----------|--------------|
| **[QUICK_START.md](QUICK_START.md)** | Get started in 5 minutes | Developers | 5 min |
| **[API_REFERENCE.md](API_REFERENCE.md)** | API endpoints & integration | Developers | 15 min |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design & architecture | Architects, Developers | 20 min |
| **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** | AWS deployment guide | DevOps, Developers | 30 min |
| **[ROI_ANALYSIS.md](ROI_ANALYSIS.md)** | Business value & ROI | Business, Executives | 10 min |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues & fixes | All | As needed |
## Documentation Structure
### For Developers
1. **Start Here**: [QUICK_START.md](QUICK_START.md)
- 5-minute setup
- Run demo
- First API call
2. **Integration**: [API_REFERENCE.md](API_REFERENCE.md)
- All endpoints (POST /predict, GET /health)
- Request/response formats
- Code examples (Python, JavaScript, Java)
- Error handling
- Best practices
3. **Understanding the System**: [ARCHITECTURE.md](ARCHITECTURE.md)
- Dual-track architecture
- Fast Lane vs Deep Lane
- Data flow
- Component diagram
- Technology stack
4. **Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- AWS serverless setup
- SAM deployment
- Model upload
- Entity risk loading
- Monitoring & alerts
5. **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Common errors & fixes
- Debug tools
- Performance optimization
- FAQ
### For Business Stakeholders
1. **Start Here**: [ROI_ANALYSIS.md](ROI_ANALYSIS.md)
- Annual savings: $61.2M
- ROI: 611,789%
- Payback period: 1.5 days
- Fraud reduction: 62.5%
- False positive reduction: 85%
2. **System Overview**: [ARCHITECTURE.md](ARCHITECTURE.md)
- High-level architecture
- Key features
- Performance metrics
- Scalability
### For DevOps
1. **Deployment**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Infrastructure as Code (SAM)
- Step-by-step deployment
- Monitoring setup
- Cost optimization
2. **Operations**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Health checks
- Log monitoring
- Performance tuning
- Scaling
## Document Descriptions
### QUICK_START.md
**Purpose**: Get started with the system in 5 minutes
**Contents**:
- Prerequisites
- Demo setup (60 seconds)
- First API call
- Next steps
**When to use**: First-time users, quick demos
---
### API_REFERENCE.md
**Purpose**: Complete API documentation for integration
**Contents**:
- Base URL & authentication
- POST /predict endpoint (fraud scoring)
- GET /health endpoint
- Request/response formats
- Reason codes (13+ codes)
- Integration examples (Python, JS, Java)
- Error handling
- Rate limits
- Best practices
**When to use**: Integrating with existing systems
---
### ARCHITECTURE.md
**Purpose**: Understand system design and architecture
**Contents**:
- Dual-track architecture diagram
- Fast Lane (Credit Card dataset, Logistic Regression)
- Deep Lane (IEEE-CIS dataset, XGBoost + Autoencoder)
- Component interactions
- Data flow
- Technology stack
- Performance metrics
- Scalability considerations
**When to use**: System design, technical discussions, onboarding
---
### DEPLOYMENT_GUIDE.md
**Purpose**: Deploy system to AWS
**Contents**:
- Prerequisites (AWS account, CLI, SAM)
- Infrastructure as Code (template.yaml)
- Step-by-step deployment (30 minutes)
- Model upload to S3
- Entity risk to DynamoDB
- API testing
- Monitoring setup
- Cost optimization
- Security best practices
**When to use**: Production deployment, staging setup
---
### ROI_ANALYSIS.md
**Purpose**: Quantify business value and ROI
**Contents**:
- Annual savings: $61.2M (conservative) to $84.9M (optimistic)
- ROI: 611,789%
- Payback period: 1.5 days
- Fraud losses reduced: 62.5% ($7M -> $2.6M/month)
- False positives reduced: 85%
- Investigation costs reduced: 67.3%
- Customer experience improvement
- Detailed calculations & assumptions
**When to use**: Business case, executive presentations, budget approval
---
### TROUBLESHOOTING.md
**Purpose**: Solve common issues and optimize performance
**Contents**:
- Quick diagnostics
- Common issues (Demo, AWS, API, Models, Data)
- Debug tools
- Performance benchmarks
- Useful commands
- Contact support
**When to use**: When encountering errors or performance issues
## Additional Resources
### In Parent Directory (`../`)
- **README.md**: Main project overview
- **DEMO/README.md**: Demo-specific instructions
- **SOURCE/**: Source code with inline comments
### External Links
- **AWS Documentation**: https://docs.aws.amazon.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **SHAP Documentation**: https://shap.readthedocs.io/
## Quick Command Reference
```bash
# Try the demo
cd ../DEMO
./run_demo.sh # or run_demo.bat on Windows
# Train models
cd ../SOURCE/scripts
python train_fast_lane.py
python train_deep_lane.py
# Deploy to AWS
cd ../SOURCE/aws
sam build && sam deploy --guided
# Test API
curl https://your-api-url.amazonaws.com/prod/health
# View logs
aws logs tail /aws/lambda/vpbank-fraud-scoring --follow
```
## Document Versions
All documentation is for **VPBank StreamGuard v1.0** (2025-11-10)
## Contributing to Docs
If you find errors or have suggestions:
1. Note the document and section
2. Describe the issue or suggestion
3. Contact the maintainer
## Support
For documentation-related questions:
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first
- Review relevant documentation section
- Contact technical support with specific questions
---
**Start with [QUICK_START.md](QUICK_START.md) to get up and running in 5 minutes!**
