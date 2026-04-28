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
from sqlalchemy.dialects.postgresql import insert

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

# --- CONFIGURATION ---
db_url = os.environ.get("DATABASE_URL")

if not db_url:
    raise ValueError("🚨 ERROR: DATABASE_URL not found! Make sure it is set in GitHub Secrets.")

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

ENGINE = create_engine(db_url)
TICKERS = ['AAPL', 'MSFT', 'JPM', 'GS']

# --- PHASE 1: EXTRACTION ---

def fetch_market_data(tickers):
    """Fetches market data and calculates baseline risk metrics."""
    print("Fetching Market Data...")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    
    appended_data = []
    
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = df.reset_index()
        
        df['ticker'] = ticker
        df['daily_return'] = df['Close'].pct_change()
        df['rolling_volatility'] = df['daily_return'].rolling(window=20).std() 
        
        mean_return = df['daily_return'].mean()
        std_return = df['daily_return'].std()
        df['price_anomaly_flag'] = abs(df['daily_return'] - mean_return) > (3 * std_return)
        
        df = df[['Date', 'ticker', 'Close', 'daily_return', 'rolling_volatility', 'price_anomaly_flag']]
        df.columns = ['date', 'ticker', 'close_price', 'daily_return', 'rolling_volatility', 'price_anomaly_flag']
        df = df.dropna()
        appended_data.append(df)
        
    return pd.concat(appended_data)

def fetch_sentiment_data(tickers):
    """Fetches news, scores sentiment, and saves raw headlines to DB."""
    print("Fetching News and Calculating Sentiment...")
    sia = SentimentIntensityAnalyzer()
    sentiment_list = []
    
    for ticker in tickers:
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            
            for item in root.findall('.//item'):
                headline = item.find('title').text
                pub_date_raw = item.find('pubDate').text
                
                try:
                    parsed_date = datetime.strptime(pub_date_raw[:25].strip(), "%a, %d %b %Y %H:%M:%S")
                    pub_date = parsed_date.strftime('%Y-%m-%d')
                except Exception:
                    pub_date = datetime.today().strftime('%Y-%m-%d')
                    
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
        # Load the raw headlines to the sentiment table just like before
        sentiment_df.to_sql('sentiment_data', ENGINE, if_exists='append', index=False)
        print(f"Successfully loaded {len(sentiment_df)} raw sentiment records to Database.")
        return sentiment_df
    else:
        print("No sentiment records found.")
        return pd.DataFrame()


# --- PHASE 2: CORRELATION ENGINE & FINAL LOAD ---

def process_and_load_master_data(df_market, df_sentiment):
    """Merges datasets, calculates Pearson correlation, and loads to DB using UPSERT."""
    print("Initiating Phase 2: Statistical Correlation Validation...")
    
    df_market['date'] = pd.to_datetime(df_market['date'])
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
    
    daily_sentiment = df_sentiment.groupby(['date', 'ticker'])['sentiment_score'].mean().reset_index()
    
    df_master = pd.merge(df_market, daily_sentiment, on=['date', 'ticker'], how='left')
    df_master['sentiment_score'] = df_master['sentiment_score'].fillna(0) 
    
    # [FIXED WARNING]: Added include_groups=False to silence the pandas deprecation warning
    df_master['sentiment_price_corr'] = df_master.groupby('ticker').apply(
        lambda x: x['sentiment_score'].rolling(window=20).corr(x['daily_return']),
        include_groups=False
    ).reset_index(level=0, drop=True)
    
    df_master['sentiment_price_corr'] = df_master['sentiment_price_corr'].fillna(0)
    df_master = df_master.drop(columns=['sentiment_score'])
    
    # --- THE UPSERT FUNCTION ---
    def postgres_upsert(table, conn, keys, data_iter):
        """Custom Postgres method to INSERT, or UPDATE if row already exists."""
        data = [dict(zip(keys, row)) for row in data_iter]
        insert_stmt = insert(table.table).values(data)
        
        # If there is a conflict on Date & Ticker, update all the other columns with the newest math
        update_dict = {c.name: c for c in insert_stmt.excluded if c.name not in ['date', 'ticker', 'id']}
        
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=['date', 'ticker'], # This is your UNIQUE constraint!
            set_=update_dict
        )
        conn.execute(upsert_stmt)
    # ---------------------------

    # Final Database Load (Now using the custom Upsert method!)
    df_master.to_sql('market_risk_data', ENGINE, if_exists='append', index=False, method=postgres_upsert)
    print(f"Successfully UPSERTED {len(df_master)} master risk/correlation records to Database.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting Alpha-Guard ETL Pipeline v1.1...")
    
    # Step 1 & 2: Extract
    market_df = fetch_market_data(TICKERS)
    sentiment_df = fetch_sentiment_data(TICKERS)
    
    # Step 3: Transform & Load
    if not market_df.empty and not sentiment_df.empty:
        process_and_load_master_data(market_df, sentiment_df)
    else:
        print("Missing data: Skipping Phase 2 correlation engine.")
        
    print("Pipeline Execution Complete! 🎉")
