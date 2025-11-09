# Quick Start Guide - VPBank StreamGuard
## For New Chat Sessions
### Option 1: Full Context (Recommended)

```
I'm working on VPBank StreamGuard fraud detection system.
Context files:
- @PROMPT.md - Full context initialization
- @MEMORY.md - Current status and progress
- @Workflow.md - Architecture and strategy
Current task: [Describe what you want to do]
```
### Option 2: Quick Continue

```
Continue VPBank StreamGuard development.
Read @MEMORY.md for status.
Current phase: [From MEMORY.md]
Task: [Specific task]
```
---
## Essential Files to Share

**Always include these 3 files in new chats:**
1. **PROMPT.md** - Context initialization (this is the master context file)
2. **MEMORY.md** - Current status, completed work, next steps
3. **Workflow.md** - Overall architecture and strategy
**Optional (for specific tasks):**
- **fraud_detection_datasets_analysis.md** - Dataset details
- **config/config.yaml** - Configuration reference
- Specific source files you're working on
---
## Current Status (2025-11-08)

- **Phase**: Day 3-4 of 14 (45% complete)
**Just Completed** :
- Fast Lane baseline training (Logistic Regression production-ready)
- Deep Lane code development (1,130+ lines)
- Critical bug fix (XGBoost object dtype error)
- Production enhancements (error handling, logging, memory)
**Next Immediate Action** :
```bash
# 1. Verify preprocessing fix
python scripts/verify_dtype_fix.py
# 2. Run Deep Lane training
python scripts/train_deep_lane.py
```
- **Expected Output**:
- XGBoost AUC: 0.90-0.95
- Entity risk CSVs for 7 entity types
- Training time: 5-10 minutes (sample mode)
---
## Common Commands
### Training

```bash
# Fast Lane (already trained)
python scripts/train_fast_lane.py
# Deep Lane (ready to run)
python scripts/train_deep_lane.py
# Debug preprocessing
python scripts/debug_ieee_dtypes.py
python scripts/verify_dtype_fix.py
```
### Verification

```bash
# Check training results
cat models/fast_lane_baseline_*/results.json
cat models/deep_lane_*/results.json
# View feature importance
cat models/deep_lane_*/xgboost_feature_importance.csv
# View entity risk
cat models/deep_lane_*/entity_risk_combined.csv
```
---
## Documentation Hierarchy

```
PROMPT.md ← START HERE (context initialization)
↓
MEMORY.md ← Current status, history, next steps
↓
Workflow.md ← Architecture, strategy, roadmap
↓
fraud_detection_datasets_analysis.md ← Dataset details
↓
Source code files ← Implementation details
```
---
## Checklist for New Sessions

When starting a new chat:
- [ ] Share PROMPT.md, MEMORY.md, Workflow.md
- [ ] State current phase from MEMORY.md
- [ ] Mention specific task/question
- [ ] Reference relevant source files if working on specific code
---
## Key Context Points (From PROMPT.md)

- **Architecture**: Dual-track (Fast Lane + Deep Lane)
- **Fast Lane**: Logistic Regression on Credit Card (PRODUCTION READY)
- **Deep Lane**: XGBoost + Autoencoder on IEEE-CIS (CODE READY)
- **Critical Fix**: Object dtype conversion in prepare_for_training()
- **Next Phase**: Execute Deep Lane training, then Feature Store (Day 6-7)
---
**Quick Reference**: See PROMPT.md for complete details
- **Last Updated**: 2025-11-08 Session 3
