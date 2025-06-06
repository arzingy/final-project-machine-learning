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

# --- Sidebar for inputs ---
with st.sidebar:
    st.header("Settings")
    dark_mode = st.checkbox("🌗 Enable Dark Mode", value=True)
    tckr = st.text_input("🔍 Stock Ticker (e.g., AAPL, MSFT, TSLA)", value="TGT").upper()

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

        h1, h2, h3, h4, h5, h6 {{
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

        # --- Prediction and Evaluation ---
        y_pred = model.predict(X_test).flatten()
        r2 = r2_score(y_test, y_pred)
        st.success(f"✅ Model Trained — R² Score: {r2:.4f}")

        # --- Visualization ---
        st.markdown("""
        <div style='background-color: #2b2b2b; padding: 15px; border-radius: 10px;'>
        <h3 style='color:#00bfff; margin: 0;'>📊 Actual vs Predicted High Prices</h3>
        </div>
        """, unsafe_allow_html=True)
        plt.style.use("seaborn-v0_8-darkgrid" if dark_mode else "seaborn-v0_8")
        df_plot = pd.DataFrame({
        'Index': np.arange(len(y_test)),
        'Actual': y_test.values,
        'Predicted': y_pred })

        fig1 = px.line(
        df_plot,
        x='Index',
        y=['Actual', 'Predicted'],
        labels={'value': 'Stock Price', 'Index': 'Test Index', 'variable': 'Legend'},
        title='Actual vs Predicted High Prices'
        )
        st.plotly_chart(fig1, use_container_width=True)

                # --- Last 30 Days Comparison (Matplotlib) ---
        st.markdown("### 📆 Last 30 Days: Actual vs Predicted High Prices")
        last_n = 30
        y_test_last = y_test[-last_n:]
        y_pred_last = y_pred[-last_n:]

        if isinstance(y_test.index, pd.DatetimeIndex):
            test_dates = y_test.index[-last_n:]
        else:
            test_dates = pd.date_range(end=pd.Timestamp.today(), periods=last_n)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(test_dates, y_test_last.values, label='Actual Price', marker='o', color='blue')
        ax3.plot(test_dates, y_pred_last, label='Predicted Price', marker='o', color='orange')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price ($)')
        ax3.set_title(f'{tckr} — Last {last_n} Days: Actual vs Predicted')
        ax3.legend()
        ax3.grid(True)
        fig3.autofmt_xdate()

        st.pyplot(fig3)



        # --- Forecast Metrics ---
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


        # --- Logging ---
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

            st.subheader("📅 Prediction History (Last 30)")
            st.dataframe(filtered_df.head(30))

            # Sidebar checkbox to toggle the graph
        show_r2_graph = st.sidebar.checkbox("Show R² Score Over Time")

    if show_r2_graph:
        st.subheader("📈 R² Score Over Time")

        # Convert 'Saved' to datetime (if not already)
        filtered_df['Saved'] = pd.to_datetime(filtered_df['Saved'])

        # Create the plot
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(filtered_df['Saved'], filtered_df['R2 Score'], marker='o', linestyle='-', color='darkblue')
    
        # Format the x-axis as datetime
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        fig3.autofmt_xdate()

        ax3.set_xlabel('Saved (Date & Time)')
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Score Over Time')
        ax3.grid(True)

        st.pyplot(fig3)



# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) and [Yahoo Finance](https://finance.yahoo.com/).")
