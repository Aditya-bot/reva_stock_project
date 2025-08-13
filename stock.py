
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from transformers import pipeline
from newsapi import NewsApiClient
import torch
import datetime
import gradio as gr
import io
import sys

# Initialize FinBERT sentiment analysis
finbert = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone')

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='')  # Replace with your actual key

def run_stock_prediction(ticker, period_choice):
    ticker = ticker.upper()
    period_choice = period_choice.upper()

    period_map = {
        '1Y': 365,
        '3Y': 365*3,
        '5Y': 365*5,
        '10Y': 365*10
    }

    if period_choice not in period_map:
        period_choice = '3Y'

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=period_map[period_choice])

    def get_stock_data(ticker, start_date, end_date):
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        df.reset_index(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() for col in df.columns.values]

        df.rename(columns={
            f'Open {ticker}': 'Open',
            f'High {ticker}': 'High',
            f'Low {ticker}': 'Low',
            f'Close {ticker}': 'Close',
            f'Adj Close {ticker}': 'Adj Close',
            f'Volume {ticker}': 'Volume',
            'Date ': 'Date',
            'Date': 'Date'
        }, inplace=True)

        return df

    stock_data = get_stock_data(ticker, start_date, end_date)
    stock_data = stock_data[['Close']]

    def prepare_data(data, look_back=90):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm, scaler

    X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm, scaler = prepare_data(stock_data)

    rf_model = RandomForestRegressor(n_estimators=500, max_depth=25, min_samples_split=4, min_samples_leaf=2, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    xgb_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=7,
        subsample=0.95,
        colsample_bytree=0.95,
        gamma=0.05,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    def build_bilstm_model(input_shape):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(units=128, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    bilstm_model = build_bilstm_model((X_train_lstm.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
    bilstm_model.fit(X_train_lstm, y_train, epochs=60, batch_size=32, callbacks=[early_stop])
    bilstm_pred = bilstm_model.predict(X_test_lstm)

    def get_finbert_sentiment(ticker):
        try:
            articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=100)['articles']
            headlines = [article['title'] for article in articles]

            if not headlines:
                return 0.0, []

            results = finbert(headlines)
            sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            scores = [sentiment_map[r['label']] * r['score'] for r in results]
            avg_sentiment = np.mean(scores)
            return avg_sentiment, list(zip(headlines[:3], results[:3]))

        except Exception:
            return 0.0, []

    sentiment_score, headlines = get_finbert_sentiment(ticker)

    def combine_predictions(rf_pred, xgb_pred, bilstm_pred, sentiment_score):
        meta_X = np.stack([rf_pred, xgb_pred, bilstm_pred.flatten()], axis=1)
        meta_model = Ridge()
        meta_model.fit(meta_X, y_test)
        base_pred = meta_model.predict(meta_X)
        sentiment_factor = 1 + (sentiment_score * 0.02)
        return base_pred * sentiment_factor

    final_pred = combine_predictions(rf_pred, xgb_pred, bilstm_pred, sentiment_score)

    def inverse_transform(scaler, data):
        return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    y_test_orig = inverse_transform(scaler, y_test)
    rf_pred_orig = inverse_transform(scaler, rf_pred)
    xgb_pred_orig = inverse_transform(scaler, xgb_pred)
    bilstm_pred_orig = inverse_transform(scaler, bilstm_pred.flatten())
    final_pred_orig = inverse_transform(scaler, final_pred)

    rf_rmse = np.sqrt(mean_squared_error(y_test_orig, rf_pred_orig))
    xgb_rmse = np.sqrt(mean_squared_error(y_test_orig, xgb_pred_orig))
    bilstm_rmse = np.sqrt(mean_squared_error(y_test_orig, bilstm_pred_orig))
    final_rmse = np.sqrt(mean_squared_error(y_test_orig, final_pred_orig))

    output = io.StringIO()
    sys.stdout = output

    print(f"\nModel Performance (RMSE - Lower is better):")
    print(f"Random Forest: {rf_rmse:.2f}")
    print(f"XGBoost: {xgb_rmse:.2f}")
    print(f"Bi-LSTM: {bilstm_rmse:.2f}")
    print(f"Combined Model: {final_rmse:.2f}")

    print(f"\nAnalyzed {ticker} with {period_choice} of historical data...")
    if headlines:
        print("\nTop headlines with sentiment:")
        for i, (headline, result) in enumerate(headlines):
            print(f"{i+1}. [{result['label']} ({result['score']:.2f})]: {headline}")

    last_price = y_test_orig[-1]
    predicted_price = final_pred_orig[-1]
    price_change = (predicted_price - last_price) / last_price * 100

    print("\n=== Stock Recommendation ===")
    print(f"Current Price: ${last_price:.2f}")
    print(f"Predicted Next Price: ${predicted_price:.2f} ({price_change:.2f}%)")
    print(f"News Sentiment: {'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'} ({sentiment_score:.2f})")

    if price_change > 5:
        print("Recommendation: STRONG BUY (Predicted increase > 5%)")
    elif price_change > 2:
        print("Recommendation: BUY (Predicted increase 2-5%)")
    elif price_change > 0:
        print("Recommendation: WEAK BUY (Predicted slight increase)")
    elif price_change < -5:
        print("Recommendation: STRONG SELL (Predicted decrease > 5%)")
    elif price_change < -2:
        print("Recommendation: SELL (Predicted decrease 2-5%)")
    elif price_change < 0:
        print("Recommendation: WEAK SELL (Predicted slight decrease)")
    else:
        print("Recommendation: HOLD (Price expected to remain stable)")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test_orig, label='Actual Price', color='black', linewidth=2)
    ax.plot(rf_pred_orig, label=f'Random Forest (RMSE: {rf_rmse:.2f})', alpha=0.7)
    ax.plot(xgb_pred_orig, label=f'XGBoost (RMSE: {xgb_rmse:.2f})', alpha=0.7)
    ax.plot(bilstm_pred_orig, label=f'Bi-LSTM (RMSE: {bilstm_rmse:.2f})', alpha=0.7)
    ax.plot(final_pred_orig, label=f'Combined Model (RMSE: {final_rmse:.2f})', color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{ticker} Stock Price Prediction ({period_choice} Historical Data)')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price ($)')
    ax.legend()
    plt.tight_layout()

    sys.stdout = sys.__stdout__
    return output.getvalue(), fig

iface = gr.Interface(
    fn=run_stock_prediction,
    inputs=[
        gr.Textbox(label="Stock Ticker (e.g., AAPL, MSFT)"),
        gr.Radio(choices=["1Y", "3Y", "5Y", "10Y"], label="Historical Data Period")
    ],
    outputs=[
        gr.Textbox(label="Model Output"),
        gr.Plot(label="Prediction Plot")
    ],
    title="Stock Price Predictor with News Sentiment",
    description="Enter a stock ticker and period. The system will predict future price and generate sentiment-based recommendation."
)

iface.launch()
