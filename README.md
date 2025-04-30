
# 📈 Stock Price Predictor with Sentiment Analysis

[![Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://python.org)
[![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey?style=for-the-badge&logo=sqlite)]
[![Sentiment Analysis](https://img.shields.io/badge/AI-Sentiment%20Analysis-brightgreen?style=for-the-badge&logo=ai)]
[![YFinance](https://img.shields.io/badge/API-YFinance-blue?style=for-the-badge)]
[![Liner Regression](https://img.shields.io/badge/ML-Linear%20Regression-brightgreen?style=for-the-badge&logo=ML)]
> A Python-powered tool that predicts **NSE stock prices** using **linear regression** and **sentiment analysis** of the latest news articles.

---

## 📌 Features

✅ Real-time stock data fetch from Yahoo Finance  
✅ Sentiment analysis of recent news using **VADER** and **GoogleNews**  
✅ Linear regression-based price forecasting  
✅ SQLite database for saving predictions  
✅ Real price check and prediction comparison  
✅ Supports any NSE stock symbol (e.g., TCS, RELIANCE, INFY)

---

## 🖼️ Screenshot (example)

*(Add your screenshots here to demonstrate UI/CLI)*

```bash
$ python predict.py
Enter NSE stock symbol (like TCS, RELIANCE): TCS

Fetching stock data...
Fetched 90 days of data for TCS.

Fetching news articles...
Analyzing sentiment...

--- Result ---
Predicted price for TCS on 2024-04-27: ₹3624.18
Live price as of 2024-04-26 15:58:00: ₹3602.44
```

---

## 🧠 How it Works

1. **Stock Data Fetching**: Uses `yfinance` to fetch 3 months of historical data.
2. **News Scraping**: Collects recent articles about the stock via `GoogleNews`.
3. **Sentiment Analysis**: Applies VADER NLP to score news sentiment.
4. **Model Training**: Builds a linear regression model using stock closing prices and sentiment.
5. **Prediction**: Forecasts the next day's closing price.
6. **Live Check**: Grabs current real-time price to compare.

---

## 📦 Dependencies

Make sure you have these installed (use `pip install`):

```bash
pandas numpy yfinance sklearn vaderSentiment GoogleNews newspaper3k nltk sqlite3
```

Also, download the VADER lexicon:

```python
import nltk
nltk.download('vader_lexicon')
```

---

## 🚀 Run the Tool

```bash
python predict.py
```

---

## 📂 Database

Predictions and real prices are stored in an SQLite database named `stock_predictions.db`.

| id | stock_name | prediction_date | predicted_price | sentiment | real_price | real_price_time |
|----|------------|-----------------|------------------|-----------|------------|------------------|

---

## ⚠️ Disclaimer

> **This tool is for educational and research purposes only.**

📉 Stock market investments involve risk.  
💡 The predictions made by this tool are **not financial advice**.  
❌ Do not use it as the sole basis for any trading or investment decisions.  
✅ Always consult with a certified financial advisor before investing.

---

## 🤝 Contributing

Pull requests are welcome. Feel free to fork the repo and suggest improvements.

