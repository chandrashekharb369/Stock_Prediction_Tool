import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
import nltk
import sqlite3
import warnings

# Initial setup
warnings.filterwarnings("ignore")
nltk.download('vader_lexicon')

# Database Setup
def init_db():
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_name TEXT,
            prediction_date TEXT,
            predicted_price REAL,
            sentiment REAL,
            real_price REAL,
            real_price_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(stock_name, date, predicted_price, sentiment, real_price, real_time):
    if isinstance(real_price, pd.Series):
        real_price = real_price.iloc[-1] if not real_price.empty else None
    if isinstance(real_price, np.float64) or isinstance(real_price, float):
        real_price = float(real_price)
    elif pd.isna(real_price):
        real_price = None

    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stock_data (stock_name, prediction_date, predicted_price, sentiment, real_price, real_price_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (stock_name, str(date), predicted_price, sentiment, real_price, real_time))
    conn.commit()
    conn.close()


def get_past_predictions_df(stock_name):
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction_date, predicted_price, sentiment FROM stock_data WHERE stock_name = ?
    ''', (stock_name,))
    rows = cursor.fetchall()
    conn.close()
    if rows:
        df = pd.DataFrame(rows, columns=['Date', 'Close', 'Sentiment'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    else:
        return pd.DataFrame(columns=['Date', 'Close', 'Sentiment'])

def fetch_stock_data(symbol, period='3mo'):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        hist.reset_index(inplace=True)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        return hist[['Date', 'Close']]
    except Exception as e:
        print(f"[Error] Could not fetch stock data: {e}")
        return None

def fetch_news(stock_name, from_date, to_date):
    googlenews = GoogleNews(start=from_date.strftime('%m/%d/%Y'), end=to_date.strftime('%m/%d/%Y'))
    googlenews.search(stock_name)
    return googlenews.result(sort=True)

def analyze_sentiment(articles):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    sentiment_dates = []
    for article in articles:
        try:
            url = article['link']
            news = Article(url)
            news.download()
            news.parse()
            text = news.text
            date = datetime.today().date()
            sentiment = analyzer.polarity_scores(text)['compound']
            sentiment_dates.append(date)
            sentiments.append(sentiment)
        except:
            continue
    min_len = min(len(sentiment_dates), len(sentiments))
    return sentiments[:min_len], sentiment_dates[:min_len]

def main():
    init_db()
    stock_name = input("Enter NSE stock symbol (like TCS, RELIANCE): ").upper()

    today = datetime.today().date()
    three_months_ago = today - timedelta(days=90)

    print("\nFetching stock data...")
    stock_data = fetch_stock_data(stock_name)
    if stock_data is None or stock_data.empty:
        print("Invalid stock symbol or no data found.")
        return
    print(f"Fetched {len(stock_data)} days of data for {stock_name}.")

    print("\nFetching news articles...")
    news = fetch_news(stock_name, three_months_ago, today)
    sentiments, sentiment_dates = analyze_sentiment(news)

    if len(sentiments) > 0:
        sentiment_df = pd.DataFrame({'Date': sentiment_dates, 'Sentiment': sentiments})
        sentiment_df = sentiment_df.groupby('Date').mean().reset_index()
        merged = pd.merge(stock_data, sentiment_df, on='Date', how='left')
        merged['Sentiment'].fillna(0, inplace=True)
    else:
        merged = stock_data.copy()
        merged['Sentiment'] = 0

    # Fetch past data from DB
    past_data = get_past_predictions_df(stock_name)
    all_data = pd.concat([merged[['Date', 'Close', 'Sentiment']], past_data], ignore_index=True)
    all_data = all_data.drop_duplicates(subset=['Date'])
    all_data = all_data.sort_values(by='Date').reset_index(drop=True)
    all_data['Day'] = np.arange(len(all_data))

    # Train model
    X = all_data[['Day', 'Sentiment']]
    y = all_data['Close']
    model = LinearRegression()
    model.fit(X, y)

    # Prediction
    next_day = len(all_data)
    today_sentiment = sentiments[-1] if len(sentiments) > 0 else 0
    predicted_price = model.predict([[next_day, today_sentiment]])[0]

    # Real-time price
    try:
        live_data = yf.download(tickers=stock_name + ".NS", period='1d', interval='1m')
        real_price = live_data['Close'].iloc[-1]
        real_time = live_data.index[-1].to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')
        time_diff_min = (datetime.now() - live_data.index[-1].to_pydatetime().replace(tzinfo=None)).seconds // 60
    except Exception as e:
        real_price = None
        real_time = "Unavailable"
        time_diff_min = "Unavailable"
        print(f"\n[Warning] Could not fetch live price: {e}")

    # Save to DB
    save_prediction(stock_name, today + timedelta(days=1), predicted_price, today_sentiment, real_price, real_time)

    print(f"\n--- Result ---")
    print(f"Predicted price for {stock_name} on {today + timedelta(days=1)}: ₹{predicted_price:.2f}")

    if isinstance(real_price, pd.Series):
        if not real_price.empty:
            last_real_price = real_price.iloc[-1]
            print(f"Live price as of {real_time}: ₹{last_real_price:.2f}")
            print(f"Time Difference: {time_diff_min} minutes")
        else:
            print("Live price unavailable (empty Series).")
    elif real_price is not None and not pd.isna(real_price):
        print(f"Live price as of {real_time}: ₹{float(real_price):.2f}")
        print(f"Time Difference: {time_diff_min} minutes")
    else:
        print("Live price unavailable.")


if __name__ == "__main__":
    main()
