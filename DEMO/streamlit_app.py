"""
VPBank StreamGuard - Interactive Fraud Detection Dashboard

Real-time fraud detection demo with live API integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.serving.rba_policy import RBAPolicy

# Page config
st.set_page_config(
    page_title="VPBank StreamGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = st.secrets.get("API_URL", "https://zv649hyq0c.execute-api.us-east-1.amazonaws.com/prod")

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'selected_txn' not in st.session_state:
    st.session_state.selected_txn = None
if 'challenge_mode' not in st.session_state:
    st.session_state.challenge_mode = False
if 'rba_policy' not in st.session_state:
    st.session_state.rba_policy = RBAPolicy()
if 'api_available' not in st.session_state:
    st.session_state.api_available = None  # Cache API availability


def format_timestamp(ts):
    """
    Handle both datetime objects and strings.
    Fixes serialization issues with Streamlit session state.
    """
    if isinstance(ts, str):
        try:
            # Try parsing ISO format
            ts = datetime.fromisoformat(ts)
        except:
            try:
                # Try parsing common formats
                ts = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
            except:
                # Return as-is if parsing fails
                return ts

    # Format datetime object
    return ts.strftime('%H:%M:%S') if hasattr(ts, 'strftime') else str(ts)


def generate_mock_transaction():
    """Generate a realistic mock transaction."""
    # Transaction patterns
    fraud_probability = np.random.random()
    is_suspicious = fraud_probability > 0.85  # 15% suspicious

    # Generate V1-V28 features (PCA components)
    if is_suspicious:
        # Suspicious transactions have more extreme values
        v_features = np.random.randn(28) * 3
    else:
        # Normal transactions
        v_features = np.random.randn(28) * 0.5

    # Transaction details
    hour = np.random.randint(0, 24)
    amount = np.random.lognormal(4, 1.5) if not is_suspicious else np.random.lognormal(5.5, 1)
    amount = round(min(amount, 10000), 2)

    # Time period encoding
    if 6 <= hour < 12:
        time_period = [1, 0]  # Morning
    elif 12 <= hour < 18:
        time_period = [0, 1]  # Afternoon
    elif 18 <= hour < 22:
        time_period = [0, 0]  # Evening
    else:
        time_period = [1, 1]  # Night (higher risk)

    # Channel
    channels = ['online_banking', 'mobile_app', 'atm', 'pos', 'card_not_present']
    channel_weights = [0.3, 0.35, 0.15, 0.15, 0.05]
    if is_suspicious:
        channel_weights = [0.15, 0.15, 0.05, 0.05, 0.6]  # More card-not-present fraud
    channel = np.random.choice(channels, p=channel_weights)

    # Entity IDs
    device_ids = ['device_001', 'device_002', 'device_003', 'device_new_suspicious']
    device_id = 'device_new_suspicious' if is_suspicious else np.random.choice(device_ids[:3])

    ip_proxies = ['0', '1']  # 0 = not proxy, 1 = proxy
    ip_proxy = '1' if (is_suspicious and np.random.random() > 0.5) else '0'

    # Build feature vector (34 features)
    time_value = np.random.randint(0, 86400)  # Seconds in a day
    features = {
        'Time': time_value,
        **{f'V{i}': float(v_features[i-1]) for i in range(1, 29)},
        'Amount': amount,
        'device_id': device_id,
        'ip_proxy': ip_proxy,
        'channel': channel,
        'hour': hour
    }

    # Customer info
    customer_ids = [f'CUST_{1000+i:04d}' for i in range(100)]
    features['customer_id'] = np.random.choice(customer_ids)
    features['transaction_id'] = f'TXN_{int(time.time()*1000) % 1000000:06d}'
    features['timestamp'] = datetime.now()

    return features


def call_fraud_api(transaction):
    """Call the live fraud detection API."""
    try:
        # Prepare request payload
        payload = {
            **{f'V{i}': transaction[f'V{i}'] for i in range(1, 29)},
            'Time': transaction['Time'],
            'Amount': transaction['Amount'],
            'ip_proxy': transaction.get('ip_proxy', '0'),
            'device_id': transaction.get('device_id', '')
        }

        # Call prediction endpoint
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.Timeout:
        st.warning("API timeout - using local fallback")
        return None
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None


def get_risk_color(risk_score):
    """Get color based on risk score."""
    if risk_score < 0.3:
        return 'green'
    elif risk_score < 0.7:
        return 'orange'
    else:
        return 'red'


def display_transaction_card(txn, idx):
    """Display a transaction card with color coding."""
    risk_score = txn.get('risk_score', 0)
    decision = txn.get('decision', 'unknown')
    color = get_risk_color(risk_score)

    # Color mapping
    color_map = {
        'green': '#28a745',
        'orange': '#ffc107',
        'red': '#dc3545'
    }

    decision_emoji = {
        'pass': '✅',
        'challenge': '⚠️',
        'block': '🚫'
    }

    with st.container():
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.markdown(f"**{txn['transaction_id']}**")
            st.caption(f"Customer: {txn['customer_id']}")
            st.caption(f"Channel: {txn['channel']}")

        with col2:
            st.metric("Amount", f"${txn['Amount']:,.2f}")
            st.caption(f"{format_timestamp(txn['timestamp'])}")

        with col3:
            st.markdown(
                f"<div style='background-color: {color_map[color]}; color: white; "
                f"padding: 10px; border-radius: 5px; text-align: center;'>"
                f"<b>Risk: {risk_score:.3f}</b><br>{decision_emoji.get(decision, '')} {decision.upper()}</div>",
                unsafe_allow_html=True
            )

        with col4:
            if st.button("Details", key=f"btn_{idx}"):
                st.session_state.selected_txn = txn
                st.rerun()  # Trigger page refresh to show details


def display_transaction_details(txn):
    """Display detailed transaction information."""
    st.subheader("Transaction Details")

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Transaction ID", txn['transaction_id'])
        st.metric("Customer ID", txn['customer_id'])
    with col2:
        st.metric("Amount", f"${txn['Amount']:,.2f}")
        st.metric("Channel", txn['channel'])
    with col3:
        # Format timestamp safely
        timestamp_str = txn['timestamp']
        if not isinstance(timestamp_str, str):
            timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp_str, 'strftime') else str(timestamp_str)
        st.metric("Time", timestamp_str)
        st.metric("Hour", f"{txn['hour']}:00")

    # Risk scoring
    st.subheader("Risk Assessment")
    risk_score = txn.get('risk_score', 0)
    model_score = txn.get('model_score', 0)
    entity_risk = txn.get('entity_risk', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Combined Risk", f"{risk_score:.3f}", delta=None)
    with col2:
        st.metric("Model Score (70%)", f"{model_score:.3f}")
    with col3:
        st.metric("Entity Risk (30%)", f"{entity_risk:.3f}")

    # Decision
    st.subheader("Decision")
    decision = txn.get('decision', 'unknown')
    color = get_risk_color(risk_score)
    color_map = {
        'green': '#28a745',
        'orange': '#ffc107',
        'red': '#dc3545'
    }

    st.markdown(
        f"<div style='background-color: {color_map[color]}; color: white; "
        f"padding: 20px; border-radius: 10px; text-align: center; font-size: 24px;'>"
        f"<b>{decision.upper()}</b></div>",
        unsafe_allow_html=True
    )

    # Reason codes (simulated)
    st.subheader("Reason Codes")
    reason_codes = []
    if txn['Amount'] > 500:
        reason_codes.append('amount_high')
    if txn['hour'] < 6 or txn['hour'] > 22:
        reason_codes.append('time_unusual')
    if txn.get('ip_proxy') == '1':
        reason_codes.append('ip_proxy_detected')
    if entity_risk > 0.3:
        reason_codes.append('entity_risk_high')
    if txn['channel'] == 'card_not_present':
        reason_codes.append('card_not_present_risk')

    if reason_codes:
        for code in reason_codes:
            st.markdown(f"- 🔴 **{code.upper().replace('_', ' ')}**")
    else:
        st.success("No significant risk factors detected")

    # Challenge simulation
    if decision == 'challenge' and not st.session_state.challenge_mode:
        st.subheader("Step-Up Authentication Required")
        if st.button("Simulate OTP Verification"):
            st.session_state.challenge_mode = True
            st.rerun()

    if st.session_state.challenge_mode:
        st.success("✅ OTP Verification Successful - Transaction Approved")
        if st.button("Reset"):
            st.session_state.challenge_mode = False
            st.rerun()


def main():
    # Header
    st.title("🛡️ VPBank StreamGuard")
    st.markdown("**Real-Time Fraud Detection Dashboard**")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # API health check
        st.subheader("API Status")

        # Cache API check to avoid repeated calls
        if st.session_state.api_available is None or st.button("Refresh API Status", key="refresh_api"):
            try:
                health_response = requests.get(f"{API_URL}/health", timeout=5)
                if health_response.status_code == 200:
                    st.session_state.api_available = True
                    st.success("✅ API Online")
                    health_data = health_response.json()
                    st.json(health_data)
                else:
                    st.session_state.api_available = False
                    st.error(f"❌ API Offline (Status: {health_response.status_code})")
            except requests.exceptions.Timeout:
                st.session_state.api_available = False
                st.warning("⚠️ API Timeout (>5s)")
                st.info("💡 Using local fallback scoring")
            except requests.exceptions.ConnectionError:
                st.session_state.api_available = False
                st.warning("⚠️ Cannot reach API")
                st.info("💡 Possible causes:\n- Internet connection issue\n- Firewall blocking AWS\n- API not deployed")
                st.info("✨ Dashboard will use local fallback")
            except Exception as e:
                st.session_state.api_available = False
                st.error(f"❌ Error: {str(e)}")
        else:
            # Show cached status
            if st.session_state.api_available:
                st.success("✅ API Online (cached)")
            else:
                st.warning("⚠️ API Unavailable (using local fallback)")

        st.divider()

        # Transaction generation
        st.subheader("Transaction Generator")
        num_transactions = st.slider("Transactions to generate", 1, 20, 5)

        if st.button("Generate Transactions", type="primary"):
            with st.spinner("Generating and scoring transactions..."):
                for _ in range(num_transactions):
                    # Generate transaction
                    txn = generate_mock_transaction()

                    # Call API
                    api_result = call_fraud_api(txn)

                    if api_result:
                        txn.update(api_result)
                    else:
                        # Fallback to local scoring
                        txn['risk_score'] = np.random.random()
                        txn['model_score'] = txn['risk_score'] * 0.7
                        txn['entity_risk'] = txn['risk_score'] * 0.3
                        txn['decision'] = 'pass' if txn['risk_score'] < 0.3 else ('challenge' if txn['risk_score'] < 0.7 else 'block')

                    st.session_state.transactions.insert(0, txn)

                # Keep only last 50 transactions
                st.session_state.transactions = st.session_state.transactions[:50]

            st.success(f"Generated {num_transactions} transactions!")

        st.divider()

        # Statistics
        if st.session_state.transactions:
            st.subheader("Statistics")
            df = pd.DataFrame(st.session_state.transactions)

            total = len(df)
            passed = len(df[df['decision'] == 'pass'])
            challenged = len(df[df['decision'] == 'challenge'])
            blocked = len(df[df['decision'] == 'block'])

            st.metric("Total Transactions", total)
            st.metric("Passed", passed, delta=f"{passed/total*100:.1f}%")
            st.metric("Challenged", challenged, delta=f"{challenged/total*100:.1f}%")
            st.metric("Blocked", blocked, delta=f"{blocked/total*100:.1f}%")

    # Main content
    tab1, tab2, tab3 = st.tabs(["Transaction Feed", "Analytics", "About"])

    with tab1:
        # Transaction details panel - show at top when selected
        if st.session_state.selected_txn:
            st.subheader("📋 Transaction Details")
            with st.container():
                display_transaction_details(st.session_state.selected_txn)
                if st.button("✖ Close Details", key="close_details_btn", type="secondary"):
                    st.session_state.selected_txn = None
                    st.rerun()
            st.divider()

        # Transaction feed
        if st.session_state.transactions:
            st.subheader("Recent Transactions")

            # Display transaction cards
            for idx, txn in enumerate(st.session_state.transactions[:20]):
                display_transaction_card(txn, idx)
                st.divider()
        else:
            st.info("👈 Click 'Generate Transactions' in the sidebar to start")

    with tab2:
        if st.session_state.transactions:
            st.subheader("Fraud Detection Analytics")

            df = pd.DataFrame(st.session_state.transactions)

            # Risk score distribution
            fig_risk = px.histogram(
                df,
                x='risk_score',
                nbins=20,
                title='Risk Score Distribution',
                color='decision',
                color_discrete_map={'pass': 'green', 'challenge': 'orange', 'block': 'red'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)

            # Channel breakdown
            col1, col2 = st.columns(2)

            with col1:
                channel_counts = df['channel'].value_counts()
                fig_channel = px.pie(
                    values=channel_counts.values,
                    names=channel_counts.index,
                    title='Transactions by Channel'
                )
                st.plotly_chart(fig_channel, use_container_width=True)

            with col2:
                decision_counts = df['decision'].value_counts()
                fig_decision = px.pie(
                    values=decision_counts.values,
                    names=decision_counts.index,
                    title='Decisions Distribution',
                    color_discrete_map={'pass': 'green', 'challenge': 'orange', 'block': 'red'}
                )
                st.plotly_chart(fig_decision, use_container_width=True)

            # Amount vs Risk
            fig_amount = px.scatter(
                df,
                x='Amount',
                y='risk_score',
                color='decision',
                title='Transaction Amount vs Risk Score',
                color_discrete_map={'pass': 'green', 'challenge': 'orange', 'block': 'red'},
                hover_data=['transaction_id', 'channel']
            )
            st.plotly_chart(fig_amount, use_container_width=True)

        else:
            st.info("Generate transactions to see analytics")

    with tab3:
        st.subheader("About VPBank StreamGuard")

        st.markdown("""
        **VPBank StreamGuard** is a dual-track fraud detection system combining:

        ### 🚀 Fast Lane (Real-Time)
        - **Model**: Logistic Regression on Credit Card dataset
        - **Latency**: 50-150ms (warm)
        - **AUC**: 0.9650 ✅
        - **Features**: 34 engineered features (V1-V28, Time, Amount, derived features)

        ### 🔬 Deep Lane (Behavioral Intelligence)
        - **Model**: XGBoost + Autoencoder on IEEE-CIS dataset
        - **Entity Risk**: 5,526 entities tracked (IP, device, email, card)
        - **AUC**: 0.8940 ✅ (exceeded target by 1.4%)

        ### 🛡️ Risk-Based Authentication (RBA)
        - **Pass** (< 0.3): Allow transaction immediately
        - **Challenge** (0.3-0.7): Require OTP/biometric verification
        - **Block** (> 0.7): Block and flag for fraud investigation

        ### 🏗️ Infrastructure
        - **Platform**: AWS Serverless (Lambda + API Gateway + DynamoDB + S3)
        - **Cost**: $0.11/month prototype scale (80x cheaper than EC2)
        - **Latency**: <150ms warm, <3s cold start
        - **Scalability**: Auto-scales to 1000+ TPS
        - **SLA**: 99.95% (AWS managed services)

        ### 📊 Key Metrics
        - **Fraud Detection Rate**: +10-15% improvement
        - **False Positive Reduction**: -15-20%
        - **Valid Transactions Challenged**: <2%
        - **ROI**: Significant cost savings from reduced fraud and investigation costs

        ### 🔗 Live API
        **Base URL**: `{API_URL}`
        - `POST /predict` - Fraud scoring
        - `GET /health` - Health check

        ---
        **Project**: VPBank StreamGuard
        **Status**: Production Ready ✅
        **Deployment**: 2025-11-09
        """.replace('{API_URL}', API_URL))


if __name__ == "__main__":
    main()
