#Dependancies 
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
import seaborn as sns
from datetime import timedelta

# For sentiment tab
import feedparser
from newspaper import Article
from transformers import pipeline
from collections import Counter
from dateutil import parser
import nltk

nltk.download('punkt')

# --- App Setup ---
st.set_page_config(page_title="Next-Day Stock Predictor", layout="wide")

# --- Tab header---
st.markdown("""
    <div style="padding-top: 40px;"></div>

    <div style="
        background: linear-gradient(135deg, #007BFF, #0056b3);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.25);
    ">
        <h1 style="color: white; margin: 0; font-size: 28px;">
             Welcome to Your Financial Toolbox 🧰
        </h1>
        <p style="color: white; font-size: 16px; margin-top: 5px;">
            Explore predictions, simulations, and market sentiment tools.
        </p>
    </div>
""", unsafe_allow_html=True)


# --- Tabs Setup ---
tab1, tab2, tab3 = st.tabs([
    "Next-Day Stock Predictor", 
    "Monte Carlo Simulation",
    "Market Sentiment"
])

with tab1:
# --- Sidebar for inputs ---
    with st.sidebar:
        st.header("Settings")
        dark_mode = False
        tckr = st.text_input("🔍 Stock Ticker (e.g., AAPL, MSFT, TSLA)", value="TGT").upper()
        show_r2_graph = st.checkbox("Show R² Score Over Time")

        # ⚠️ Add disclaimer here
        st.markdown("""
        <hr style='margin-top: 20px; margin-bottom: 10px;'>
        <p style='font-size: 14px; color: gray;'>
        ⚠️ **Please Read Disclaimer Before Use**   
        <br><br>
        This site serves as an informational tool ONLY. It does not serve the purpose of giving personal or financial advice of any kind. Make investments at your own risk. All investing comes with a degree of risk. That includes the possible risk of Loss (Financial loss). Creators/contributors of this site are not responsible for any loss or actions taken by the user. Actions and consequences remain the full responsibility of the user.
        </p>
        """, unsafe_allow_html=True)    

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
        st.markdown("**⚠️ Please Read Disclaimer Before Use**")


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
                    'Actual': '#1f77b4',
                    'Predicted': "#ccabda"
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
            
            last_n = 30
            y_test_last = y_test[-last_n:]
            y_pred_last = y_pred[-last_n:]

            df_plot = pd.DataFrame({
                'Date': y_test_last.index,
                'Actual': y_test_last.values,
                'Predicted': y_pred_last
            })

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
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    ax3.plot(filtered_df['Saved'], filtered_df['R2 Score'], marker='o')
                    ax3.set_title(f"R² Score Over Time for {tckr}")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("R² Score")
                    plt.xticks(rotation=45)
                    st.pyplot(fig3)

# ---------------------------------------------------
# --- Tab 2: Monte Carlo Simulation code below ---
# ---------------------------------------------------

with tab2:

    st.markdown("""
    <h2 style='
        color: #0066cc;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: Merriweather, serif;
        margin-bottom: 1rem;'>
        In it for the Long Haul? 🧐
    </h2>
    """, unsafe_allow_html=True)
    
    # Input ticker in tab 2
    ticker_mc = st.text_input("Enter Stock Ticker for Monte Carlo Simulation:", value="TGT").upper()

    num_simulations = st.slider("Number of simulations:", 100, 5000, 1000, 100)

    def plot_histogram(simulations, ticker, t):
        final_prices = simulations[:, -1]  # Extract final prices from all simulations

        prediction_date = (datetime.today() + timedelta(days=t)).strftime("%B %d, %Y")

        p5, p50, p95 = np.percentile(final_prices, [5, 50, 95])

        st.write(f"Monte Carlo Simulation for {ticker} - Projected Price on {prediction_date}:")
        st.write(f"5th Percentile (Low Risk Estimate): ${p5:.2f}")
        st.write(f"Median Price (Most Likely Outcome): ${p50:.2f}")
        st.write(f"95th Percentile (High Reward Estimate): ${p95:.2f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(final_prices, bins=50, kde=True, color="blue", alpha=0.6, ax=ax)

        ax.axvline(p5, color="red", linestyle="dashed", label="5th Percentile (Low Risk)")
        ax.axvline(p50, color="black", linestyle="dashed", label="Median Price (Most Likely Outcome)")
        ax.axvline(p95, color="green", linestyle="dashed", label="95th Percentile (High Reward)")

        ax.set_title(f"Probability Distribution of {ticker} Stock Price Over {t} Days ({prediction_date})")
        ax.set_xlabel("Projected Stock Price, $")
        ax.set_ylabel("Frequency")
        ax.legend()

        st.pyplot(fig)

    def plot_paths(simulations, historical_prices, ticker, t):
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(simulations.T, alpha=0.05, color="blue")

        ax.plot(range(len(historical_prices)), historical_prices, color="red", linewidth=2,
                label=f"Historical trajectory for last {t} days")

        ax.set_title(f"Monte Carlo Simulation with Trends for {ticker} Over {t} Days")
        ax.set_xlabel("Days")
        ax.set_ylabel("Projected Stock Price, $")
        ax.legend()

        st.pyplot(fig)

    def monte_carlo(ticker, num_simulations=1000):
        stock_data = yf.download(ticker, start='2020-01-01', end='2025-12-31')
        stock_data["Returns"] = np.log(stock_data["Close"] / stock_data["Close"].shift(1))

        mean_return = stock_data["Returns"].mean()
        volatility = stock_data["Returns"].std()
        last_price = stock_data["Close"].iloc[-1]

        time_frames = [7, 30, 90, 180, 365]

        for t in time_frames:
            simulations = np.zeros((num_simulations, t))

            for sim in range(num_simulations):
                price_series = np.zeros(t)
                price_series[0] = float(last_price.iloc[0])
                for i in range(1, t):
                    price_series[i] = price_series[i - 1] * np.exp(np.random.normal(mean_return, volatility))
                simulations[sim, :] = price_series

            historical_prices = stock_data["Close"][-t:]
            st.subheader(f"Simulation over {t} days:")
            plot_histogram(simulations, ticker, t)
            plot_paths(simulations, historical_prices, ticker, t)

    if ticker_mc:
        with st.spinner("Running Monte Carlo simulations..."):
            monte_carlo(ticker_mc, num_simulations=num_simulations)


# ---------------------------------------------------
# --- Tab 3: Article Market Sentiment ---
# ---------------------------------------------------

with tab3:

    st.markdown("""
    <h2 style='
        color: #0066cc;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: Merriweather, serif;
        margin-bottom: 1rem;'>
        📰 The VIBE of the day 😎
    </h2>
    """, unsafe_allow_html=True)

    # Load Sentiment Model (once per session)
    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis")
    
    sentiment_model = load_sentiment_model()

    # RSS Feeds
    rss_feeds = {
        "Reuters": "https://feeds.reuters.com/reuters/businessNews",
        "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html"
    }

    # Helper Functions
    def fetch_articles(feed_url, limit=5):
        feed = feedparser.parse(feed_url)
        return feed.entries[:limit]

    def extract_article_text(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return ""

    def analyze_full_article(text):
        chunk_size = 1024
        sentiments = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip()) < 20:
                continue
            result = sentiment_model(chunk)[0]
            sentiments.append(result['label'].upper())
        if not sentiments:
            return "NEUTRAL", 0
        counts = Counter(sentiments)
        label = counts.most_common(1)[0][0]
        confidence = round(counts[label] / len(sentiments), 3)
        return label, confidence

    def plot_sentiment_bar(sentiment_counts):
        negative = sentiment_counts.get("NEGATIVE", 0)
        neutral = sentiment_counts.get("NEUTRAL", 0)
        positive = sentiment_counts.get("POSITIVE", 0)
        total = negative + neutral + positive
        if total == 0:
            st.write("No sentiment data to plot.")
            return
        labels = ['Negative', 'Neutral', 'Positive']
        values = [negative, neutral, positive]
        colors = ["#690606", '#CCCCCC', "#3ADD28"]
        fig, ax = plt.subplots(figsize=(8, 1.5))
        left = 0
        for v, c, l in zip(values, colors, labels):
            ax.barh(0, v, left=left, color=c, edgecolor='black')
            if v > 0:
                ax.text(left + v/2, 0, f"{l}: {v}", ha='center', va='center', fontsize=9, color='white')
            left += v
        ax.set_yticks([])
        ax.set_xlim(0, total)
        ax.set_title("Mood Distribution", fontsize=12)
        plt.box(False)
        st.pyplot(fig)

    # UI
    if st.button("🧠 Analyze Market Mood"):
        with st.spinner("Analyzing headlines from top financial sources..."):
            total_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

            for source, url in rss_feeds.items():
                entries = fetch_articles(url)
                for entry in entries:
                    article_text = extract_article_text(entry.link)
                    if not article_text:
                        continue
                    label, _ = analyze_full_article(article_text)
                    total_counts[label] += 1

            total = sum(total_counts.values())
            if total == 0:
                st.warning("No articles could be analyzed.")
            else:
                dominant = max(total_counts, key=total_counts.get)
                label_map = {
                    "POSITIVE": "🔥 Hot Market",
                    "NEGATIVE": "🥶 Cold Market",
                    "NEUTRAL": "😐 Lukewarm Market"
                }
                st.markdown(f"### {label_map[dominant]}")
                st.caption(f"Based on {total} financial news articles today.")
                plot_sentiment_bar(total_counts)
                
            headlines = []

            for source, url in rss_feeds.items():
                entries = fetch_articles(url)
                for entry in entries:
                    article_text = extract_article_text(entry.link)
                    if not article_text:
                        continue
                    label, confidence = analyze_full_article(article_text)
                    total_counts[label] += 1
                    headlines.append((entry.title, label.title(), entry.link))

            # Sort headlines by sentiment
            sentiment_colors = {
                "Positive": "green",
                "Neutral": "gray",
                "Negative": "red"
            }

            st.markdown("### 🗞️ Latest Headlines & Sentiment")
            for title, label, link in headlines:
                st.markdown(f"<span style='color:{sentiment_colors[label]}; font-weight:bold'>{label}</span>: <a href='{link}' target='_blank'>{title}</a>", unsafe_allow_html=True)
