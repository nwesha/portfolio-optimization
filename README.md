# 📊 Quantitative Portfolio Allocation & ML Forecasting

An end-to-end Data Science and Quantitative Finance pipeline that optimizes stock portfolios using **Modern Portfolio Theory (Markowitz)** and forecasts asset returns using **Machine Learning (Random Forest)**. The system includes an AI-powered conversational interface to query the ML models and analytics in real-time.

## 🧠 Core Machine Learning & Quant Pipeline

### 1. Feature Engineering (Time-Series)
- Constructed predictive features from historical price data.
- **Momentum:** 3-month and 6-month rolling averages to capture asset trends.
- **Volatility:** 3-month and 6-month rolling standard deviations to measure risk.
- **Market Baseline:** 1-month lagged market returns to capture macroeconomic sentiment.

### 2. Walk-Forward Validation
- Implemented strict chronological train/test splitting to prevent **data leakage**.
- **Training Period:** 2019-04-01 to 2022-12-31.
- **Testing Period (Out-of-sample):** 2023-01-01 to 2025-03-31.

### 3. Return Forecasting (Random Forest)
- Trained a `RandomForestRegressor` (scikit-learn) to predict 1-month forward returns.
- Handled noisy financial data using strict hyperparameters (`max_depth=6`, `min_samples_leaf=8`) to prevent overfitting.
- **Evaluation:** Achieved a **53.09% Directional Accuracy** (predicting the correct sign of the return), which provides a tradable edge and beats the zero-return naive baseline.

### 4. Portfolio Optimization (Markowitz Mean-Variance)
- Solved the complex convex optimization problem using `cvxpy`.
- **Objective:** Maximize expected returns while penalizing variance (risk).
- **Constraints:** 
  - Fully invested (weights sum to 1).
  - Long-only (no short selling).
  - Concentration limits: Max 15% per individual stock, max 25% per sector to mathematically enforce diversification.

https://github.com/user-attachments/assets/ea28d24d-5957-4f27-8fcd-90235f5e442a

## 🚀 The App (Streamlit + LLM)

### 🌐 Live Application
The fully functional web application is deployed and available to use here:  
**[👉 Try the AI Portfolio Advisor App](https://portfolio-optimization-agent.streamlit.app)**  
*(Note: You will need a free Google Gemini API key to interact with the AI agent)*


The core ML pipeline is wrapped in a Streamlit web application. An autonomous LLM agent (powered by the Gemini API) is given access to the ML models via **Tool Calling / Function Calling**. 

Users can ask questions in natural language (e.g., *"Forecast next month's returns for the tech sector"*), and the agent will dynamically execute the Random Forest model and Markowitz optimizer to return data-backed insights.

## 📁 Project Structure

```
Portfolio-Allocation/
├── src/
│   ├── data_fetcher.py       # yfinance API integration
│   ├── risk_analysis.py      # Covariance & Sharpe ratio calculations
│   ├── optimizer.py          # Markowitz CVXPY solver
│   ├── ml_forecaster.py      # Random Forest feature engineering & training
│   └── agent.py              # LLM Tool-Calling orchestration
├── notebooks/
│   ├── Portfolio_ML_Workflow.ipynb  # Pure Data Science workflow
│   └── AI_Agent_Tutorial.ipynb      # Tool-calling tutorial
├── app.py                    # Streamlit interface
└── requirements.txt
```

## 🔧 Tech Stack
- **Data Science:** Python, pandas, numpy, scikit-learn
- **Quantitative Finance:** cvxpy, yfinance
- **Deployment & Engineering:** Streamlit, Google Generative AI (Gemini Tool Calling)

## 📊 Stock Universe
Analyzes 18 US large-cap stocks across 6 sectors (Tech, Finance, Energy, Consumer Staples, Healthcare, Industrial) to ensure diversified sector exposure during optimization.
