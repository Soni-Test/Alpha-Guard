import os
import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# --- CONFIGURATION ---
# 1. Securely fetch the Neon URL from GitHub Secrets 
db_url = os.environ.get("DATABASE_URL")

# Safety Check: Stop the script if the database URL isn't found
if not db_url:
    raise ValueError("🚨 ERROR: DATABASE_URL not found! Make sure it is set in GitHub Secrets.")

# 2. Fix the Cloud Postgres Protocol Quirk (Reminder : Neon uses postgres://, SQLAlchemy needs postgresql://)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# 3. Create the engine securely
ENGINE = create_engine(db_url)

# Tickers
TICKERS = ['AAPL', 'MSFT', 'JPM', 'GS']


# Code---
def process_market_data(tickers):
    """Fetches market data and calculates risk metrics (Volatility & Anomalies)."""
    print("Fetching Market Data...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365) # 1 year of data for rolling metrics
    
    appended_data = []
    
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = df.reset_index()
        
        # Calculate Risk Metrics
        df['ticker'] = ticker
        df['daily_return'] = df['Close'].pct_change()
        # 20-day rolling volatility (Standard Deviation of returns)
        df['rolling_volatility'] = df['daily_return'].rolling(window=20).std() 
        
        # Anomaly Detection: Flag if the daily return is an outlier 
        mean_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()
        df['price_anomaly_flag'] = abs(df['daily_return'] - mean_return) > (3 * std_return)
        
        # Clean up for DB insertion
        df = df[['Date', 'ticker', 'Close', 'daily_return', 'rolling_volatility', 'price_anomaly_flag']]
        df.columns = ['date', 'ticker', 'close_price', 'daily_return', 'rolling_volatility', 'price_anomaly_flag']
        df = df.dropna()
        appended_data.append(df)
        
    final_df = pd.concat(appended_data)
    
    # Loading to PostgreSQL
    final_df.to_sql('market_risk_data', ENGINE, if_exists='append', index=False)
    print(f"Successfully loaded {len(final_df)} market records to Database.")

def process_sentiment_data(tickers):
    """Fetches recent news via stable XML RSS feeds and applies NLP sentiment scoring."""
    print("Fetching News and Calculating Sentiment via RSS...")
    sia = SentimentIntensityAnalyzer()
    sentiment_list = []
    
    for ticker in tickers:
        # Using Yahoo's official RSS feed (much more stable than yfinance)
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        
        try:
            # Add a User-Agent header so Yahoo doesn't block our script
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            # Loop through every news item in the XML
            for item in root.findall('.//item'):
                headline = item.find('title').text
                pub_date_raw = item.find('pubDate').text
                
                # Convert RSS date format (e.g., 'Fri, 19 Apr 2026 12:00:00 +0000') to 'YYYY-MM-DD'
                try:
                    # Python 3.7+ can parse the timezone with %z, but we'll slice it to be safe
                    parsed_date = datetime.strptime(pub_date_raw[:25].strip(), "%a, %d %b %Y %H:%M:%S")
                    pub_date = parsed_date.strftime('%Y-%m-%d')
                except Exception:
                    # Fallback if date format is slightly different
                    pub_date = datetime.today().strftime('%Y-%m-%d')
                    
                # AI Sentiment Calculation (-1.0 to 1.0)
                sentiment_score = sia.polarity_scores(headline)['compound']
                
                sentiment_list.append({
                    'date': pub_date,
                    'ticker': ticker,
                    'headline': headline,
                    'sentiment_score': sentiment_score
                })
        except Exception as e:
            print(f"Failed to fetch news for {ticker}. Error: {e}")
            
    if sentiment_list:
        sentiment_df = pd.DataFrame(sentiment_list)
        
        # Load to PostgreSQL
        sentiment_df.to_sql('sentiment_data', ENGINE, if_exists='append', index=False)
        print(f"Successfully loaded {len(sentiment_df)} sentiment records to Database.")
    else:
        print("No sentiment records were loaded.")

if __name__ == "__main__":
    print("Starting Alpha-Guard ETL Pipeline...")
    process_market_data(TICKERS)
    process_sentiment_data(TICKERS)
    print("Pipeline Execution Complete!")
