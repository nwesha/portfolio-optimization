"""
AI Portfolio Advisor — Streamlit Chat Interface

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.agent import PortfolioAgent


# Page Config
st.set_page_config(
    page_title="AI Portfolio Advisor | Quant Engine",
    layout="wide",
)

# Custom Styling (Fintech / Trading Vibe)
st.markdown("""
<style>
    /* Main background - sleek dark mode like TradingView */
    .stApp {
        background-color: #0b0e14;
        color: #d1d4dc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #2a2e39;
    }
    
    /* Chat inputs */
    .stChatInputContainer {
        border: 1px solid #2962ff !important;
        border-radius: 8px !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 8px 0;
        margin-bottom: 8px;
    }
    
    /* Assistant Avatar */
    [data-testid="chatAvatarIcon-assistant"] {
        background-color: #2962ff !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Courier New', Courier, monospace;
    }

    /* Dividers */
    hr {
        border-color: #2a2e39 !important;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/bullish.png", width=60)
    st.title("Quant Engine")
    st.caption("AI-Powered Portfolio Advisory")
    
    st.divider()

    st.markdown("### API Authentication")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Required for inference. Get one at aistudio.google.com",
        placeholder="AIzaSy..."
    )
    
    st.markdown(
        "[Get your Gemini API key →](https://aistudio.google.com/api-keys)"
    )

    st.divider()
    
    st.markdown("### Market Universe")
    st.markdown("""
    <div style='font-family: monospace; font-size: 0.9em; color: #a0aec0;'>
    <b>TECH</b>: AAPL, MSFT, NVDA<br>
    <b>FIN</b>: JPM, BAC, GS<br>
    <b>ENG</b>: XOM, CVX, COP<br>
    <b>CONS</b>: PG, KO, WMT<br>
    <b>HLTH</b>: JNJ, UNH, PFE<br>
    <b>IND</b>: CAT, BA, MMM
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    st.markdown("### Quick Commands")
    if st.button("Forecast Tech Sector"):
        st.session_state.shortcut = "Forecast next-month returns for AAPL, MSFT, and NVDA."
    if st.button("Optimize Low Risk"):
        st.session_state.shortcut = "Build me a low risk conservative portfolio from the universe."


# Main Dashboard
st.markdown("## Market Terminal")

# Add a live-looking ticker strip (static metrics for the vibe)
col1, col2, col3, col4 = st.columns(4)
col1.metric("S&P 500", "5,234.18", "+1.24%")
col2.metric("NASDAQ 100", "18,399.52", "+1.58%")
col3.metric("US 10Y", "4.21%", "-0.05")
col4.metric("VIX", "13.25", "-4.12%")

st.divider()

# Chat State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "**System Initialized.**\n\n"
                "I am your Quantitative Portfolio Advisor. I have access to real-time market data, "
                "Markowitz mean-variance optimization models, and Random Forest forecasting pipelines.\n\n"
                "How can I assist with your capital allocation today?"
            ),
        }
    ]

if "agent" not in st.session_state:
    st.session_state.agent = None


# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Handle User Input
# Check if a shortcut button was pressed, otherwise take chat input
prompt = st.chat_input("Enter query (e.g., 'Compare risk profiles of NVDA and AAPL')...")
if "shortcut" in st.session_state and st.session_state.shortcut:
    prompt = st.session_state.shortcut
    st.session_state.shortcut = None

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialise agent if needed
    if st.session_state.agent is None:
        if not api_key:
            with st.chat_message("assistant"):
                err = "**Auth Error:** Please provide a valid Gemini API Key in the sidebar."
                st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()
        try:
            st.session_state.agent = PortfolioAgent(api_key=api_key)
        except Exception as e:
            with st.chat_message("assistant"):
                err = f"**System Error:** Failed to initialize agent: {e}"
                st.markdown(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Executing quantitative analysis pipeline..."):
            try:
                response = st.session_state.agent.run(prompt)
            except Exception as e:
                response = f"**Execution Error:** {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
