import streamlit as st
import plotly.express as px
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.dates as mdates

# --- App Setup ---
st.set_page_config(page_title="Next-Day Stock Predictor", layout="wide")

# --- Tabs Setup ---
tab1, tab2 = st.tabs(["Next-Day Stock Predictor", "Monte Carlo Simulation (Coming Soon)"])

with tab1:
    # --- Sidebar for inputs ---
    with st.sidebar:
        st.header("Settings")
        dark_mode = st.checkbox("🌗 Enable Dark Mode", value=True)
        tckr = st.text_input("🔍 Stock Ticker (e.g., AAPL, MSFT, TSLA)", value="TGT").upper()

        # Sidebar checkbox to toggle the R² graph
        show_r2_graph = st.checkbox("Show R² Score Over Time")

    # --- Apply dark mode styles + Merriweather font ---
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap');

            html, body, [class*="css"] {{
                font-family: 'Merriweather', serif;
                background-color: {'#1e1e1e' if dark_mode else '#f0f2f6'};
                color: {'#ffffff' if dark_mode else '#000000'};
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}

            h1 {{
                font-family: 'Merriweather', serif;
                font-weight: 700;
                color: {'#00bfff' if dark_mode else '#0066cc'};
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
            }}

            h2, h3, h4, h5, h6 {{
                font-family: 'Merriweather', serif;
                font-weight: 700;
                color: {'#00bfff' if dark_mode else '#0066cc'};
            }}
        </style>
    """, unsafe_allow_html=True)


    if tckr:
        ticker_obj = yf.Ticker(tckr)
        company_name = ticker_obj.info.get("longName", "Unknown Company")
        logo_url = ticker_obj.info.get("logo_url")

        if logo_url:
            st.image(logo_url, width=100)

        st.title(f"📈 {company_name} ({tckr}) — Next-Day Stock Price Predictor")
        st.caption("A neural network model to forecast the *next day's high price* for any S&P 500 stock.")

    if st.button("🚀 Predict Price"):
        with st.spinner("⏳ Downloading data and training model..."):

            # --- Data Preparation ---
            df = yf.download(tckr, start='2020-01-01', end='2025-12-31')
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df['Next_High'] = df['High'].shift(-1).fillna(df['Close'])

            X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            y = df['Next_High']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

            # --- Model Definition ---
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

            history = model.fit(
                X_train, y_train,
                epochs=200, batch_size=32, validation_split=0.1,
                callbacks=[early_stop, reduce_lr], verbose=0
            )

            # --- Prediction and Evaluation R² Score Output ---
            y_pred = model.predict(X_test).flatten()
            r2 = r2_score(y_test, y_pred)
            r2_percent = r2 * 100

            if r2_percent >= 80:
                st.success(f"✅ Model Trained Accuracy Score: {r2_percent:.2f}%")
            else:
                st.markdown(f"""
                <div style="
                    background-color: #ffe6e6;
                    color: #990000;
                    padding: 15px;
                    border-left: 6px solid #ff4d4d;
                    border-radius: 6px;
                    margin-bottom: 20px;">
                    <strong>❌ Model Accuracy Below Target:</strong><br>
                    Accuracy Score: {r2_percent:.2f}%
                </div>
                """, unsafe_allow_html=True)

            # --- Forecast Summary Metrics output---
            today_price = y_pred[-2]
            tomorrow_price = y_pred[-1]
            price_change = tomorrow_price - today_price

            st.markdown(f"""
            <div style="
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 6px solid #00bfff;
            border-radius: 6px;
            margin-bottom: 20px;">
            <h4 style="color: #333; margin-top: 0;">📢 Forecast Summary</h4>
            <p>Today: ${today_price:.2f}<br>
            Tomorrow: ${tomorrow_price:.2f}<br>
            Change: {price_change:+.2f}</p>
            <p style="font-weight:bold; color: {'green' if price_change > 0 else 'red' if price_change < 0 else 'gray'};">
            {"📈 Expecting an increase tomorrow." if price_change > 0 else "📉 Expecting a drop tomorrow." if price_change < 0 else "➖ No significant price movement expected."}
            </p>
            </div>
            """, unsafe_allow_html=True)

            # --- Actual vs Predicted High Prices Visualization ---
            st.markdown("""
            <div style='
                background-color: rgba(128, 128, 128, 0.3);
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.3);'>
            <h3 style='color:#004080; margin: 0;'>📊 Actual vs Predicted High Prices</h3>
            </div>
            """, unsafe_allow_html=True)

            plt.style.use("seaborn-v0_8-darkgrid" if dark_mode else "seaborn-v0_8")

            df_plot = pd.DataFrame({
                'Index': np.arange(len(y_test)),
                'Actual': y_test.values,
                'Predicted': y_pred
            })

            fig1 = px.line(
                df_plot,
                x='Index',
                y=['Actual', 'Predicted'],
                labels={'value': 'Stock Price', 'Index': 'Test Index', 'variable': 'Legend'},
                title='Actual vs Predicted High Prices',
                color_discrete_map={
                    'Actual': '#1f77b4',        # default plotly blue
                    'Predicted': "#ccabda"      # pale purple
                }
            )

            st.plotly_chart(fig1, use_container_width=True)
            
            # --- Visualization: Last 30 Days ---
            st.markdown("""
            <div style='
                background-color: rgba(128, 128, 128, 0.3);
                padding: 15px;
                border-radius: 10px;
                box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.3);'>
            <h3 style='color:#004080; margin: 0;'>📊 Actual vs Predicted High Prices (Last 30 Days)</h3>
            </div>
            """, unsafe_allow_html=True)
            

            # Slice last 30 values
            last_n = 30
            y_test_last = y_test[-last_n:]
            y_pred_last = y_pred[-last_n:]

            # Build DataFrame using real dates
            df_plot = pd.DataFrame({
                'Date': y_test_last.index,
                'Actual': y_test_last.values,
                'Predicted': y_pred_last
            })

            # Plotly line chart
            fig1 = px.line(
                df_plot,
                x='Date',
                y=['Actual', 'Predicted'],
                labels={'value': 'Stock Price ($)', 'Date': 'Date', 'variable': 'Legend'},
                title='Actual vs Predicted High Prices (Last 30 Days)',
                color_discrete_map={
                    'Actual': "#390655",     
                    'Predicted': "#8ad7ee"   
                }
            )

            # Add interactivity
            fig1.update_traces(
                mode='lines+markers',
                marker=dict(size=6),
                hovertemplate='%{y:.2f}'
            )

            fig1.update_layout(
                xaxis=dict(
                    title='Date',
                    rangeslider=dict(visible=True),
                    tickformat='%b %d',
                ),
                yaxis_title='Stock Price ($)',
                hovermode='x unified',
                plot_bgcolor='#1e1e1e' if dark_mode else '#ffffff',
                paper_bgcolor='#1e1e1e' if dark_mode else '#ffffff',
                font=dict(color='white' if dark_mode else 'black')
            )

            st.plotly_chart(fig1, use_container_width=True)

            # --- Logging Prediction ---
            save_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_df = pd.DataFrame({
                'Ticker': [tckr],
                'Actual': [y_test.iloc[-1]],
                'Predicted': [tomorrow_price],
                'R2 Score': [r2],
                'Saved': [save_time]
            }, index=[y_test.tail(1).index[0]])
            result_df.reset_index(inplace=True)
            result_df.rename(columns={'index': 'Date'}, inplace=True)

            csv_path = 'predictions_log.csv'
            if os.path.exists(csv_path):
                result_df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                result_df.to_csv(csv_path, mode='w', header=True, index=False)

            st.info(f"📥 Prediction saved at {save_time}")

            # --- History & Trend ---
            if os.path.exists(csv_path):
                df_log = pd.read_csv(csv_path)
                df_log['Saved'] = pd.to_datetime(df_log['Saved'])
                filtered_df = df_log[df_log['Ticker'] == tckr].sort_values(by='Saved', ascending=False)

                st.markdown("""
                <div style="text-align: center; padding-top: 10px;">
                    <h3 style="color:#004080;">📅 Prediction History (Last 30)</h3>
                </div>
                """, unsafe_allow_html=True)

                # Center the dataframe using a container with fixed width and horizontal margins
                st.markdown("""
                <div style="display: flex; justify-content: center;">
                    <div style="width: 80%;">
                """, unsafe_allow_html=True)

                st.dataframe(filtered_df.head(30), use_container_width=True)

                st.markdown("""
                    </div>
                </div>
                """, unsafe_allow_html=True)


                if show_r2_graph:
                    st.subheader("📈 R² Score Over Time")

                    # Create the plot
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
