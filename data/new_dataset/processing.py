from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
import re
import numpy as np
import sys
sys.path.append("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/dataset_processing/openai_analysis.py")
# from openai_analysis import get_relevance
from data.new_dataset.openai_analysis2 import RelevanceEstimator

keywords_AMZN = ["Amazon", "AMZN", "Jeff Bezos", "Bezos", "Andy Jassy", "Amazon Web Services", "AWS"]
keywords_AAPL = ["Apple", "AAPL", "IPhone", "IPad", "Macbook", "Ios", "App store", "Tim Cook"]
keywords_MSFT = ["Microsoft", "MSFT", "Windows", "Azure", "Office 365", "Satya Nadella", "Bill Gates"]
keywords_NVDA = ["Nvidia", "NVDA", "GeForce", "RTX", "Jensen Huang"]

def contains_keywords(headline, keywords):
    # Create a regex pattern for each keyword to match whole words
    if isinstance(headline, float) and np.isnan(headline):
        return False
    patterns = [re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) for keyword in keywords]
    # Check if any pattern matches the headline
    return any(pattern.search(headline) for pattern in patterns)

analyzer = SentimentIntensityAnalyzer()
estimator = RelevanceEstimator()

def get_sentiment(text):
    return analyzer.polarity_scores(text)

# Function to create batches of headlines
def batch_headlines(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]

def get_timehorizon(file_path):
    df = pd.read_csv(file_path)
    df = df[['Date', 'Headline']]
    df = df.sort_values(by='Date')
    df = df.drop_duplicates()
    # df['TH'] = df['Headline'].apply(estimator.get_relevance)
    size = df.shape[0]
    print(f"Starting {file_path}. Size if {size}")

    batch_size = 50 # Adjust this based on your needs
    all_relevances = []

    for batch in batch_headlines(df, batch_size):
        headlines = batch['Headline'].tolist()
        relevances = estimator.get_relevance(headlines, size)
        all_relevances.extend(relevances)

    df['TH'] = all_relevances
    df = df[['Date', 'Headline', 'TH']]
    df = df.sort_values(by='Date')
    df = df.drop_duplicates()
    print(f"Done with {file_path}")
    
    #df['TH'] = df['Headline'].apply(lambda headline: estimator.get_relevance(headline, size))
    return df

def merge_datasets(file1, file2, file3):
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    start_date_dt = pd.to_datetime(start_date).date()
    end_date_dt = pd.to_datetime(end_date).date()

    df_other_AAPL = pd.read_csv(file1)
    df_headlines_AAPL = pd.read_csv(file2)
    df_abc_AAPL = pd.read_csv(file3)

    merged_df = pd.merge(df_other_AAPL, df_headlines_AAPL, on='date', how='outer')
    merged_df = pd.merge(merged_df, df_abc_AAPL, on='date', how='outer')
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['date'] = merged_df['date'].dt.date
    merged_df = merged_df.sort_values('date')

    columns_to_replace = ['mean_compound_x', 'mean_compound_y', 'mean_compound']
    merged_df[columns_to_replace] = merged_df[columns_to_replace].replace(0, np.nan)
    merged_df['mean compound'] = merged_df[columns_to_replace].mean(axis=1)
    merged_df['mean compound'] = merged_df['mean compound'].replace(np.nan, 0)

    merged_df = merged_df.rename(columns={'TH': 'TH_z'})
    columns_to_replace = ['TH_x', 'TH_y', 'TH_z']
    merged_df[columns_to_replace] = merged_df[columns_to_replace].replace(0, np.nan)
    merged_df['TH'] = merged_df[columns_to_replace].mean(axis=1)
    merged_df['TH'] = merged_df['TH'].replace(np.nan, 0)

    merged_df = merged_df[['date', 'mean compound', 'TH']]

    return merged_df

def merge_sentiments(file_path_news, file_path_twitter, file_path_stocks):
    start_date = '2015-01-01'
    end_date = '2020-01-01'
    start_date_dt = pd.to_datetime(start_date).date()
    end_date_dt = pd.to_datetime(end_date).date()
    data_news = pd.read_csv(file_path_news)
    data_twitter = pd.read_csv(file_path_twitter)

    merged_df = pd.merge(data_news, data_twitter, on='date', how='outer')
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['date'] = merged_df['date'].dt.date
    merged_df = merged_df.sort_values('date')
    merged_df.fillna(0, inplace=True)

    data_stock = pd.read_csv(file_path_stocks)

    data_stock = data_stock.rename(columns={'Date': 'date'})
    data_stock['date'] = pd.to_datetime(data_stock['date'])
    data_stock['date'] = data_stock['date'].dt.date
    data_stock = data_stock[~((data_stock['date'] >= end_date_dt))]
    data_stock = data_stock[~((data_stock['date'] <= start_date_dt))]
    data_stock = data_stock[['date', 'Close', 'Volume', 'Open', 'High', 'Low', 'Adj Close']]

    merged_df = pd.merge(merged_df, data_stock, on='date', how='outer', suffixes=('_df_merged', 'df_amzn'))
    merged_df = merged_df.sort_values('date')
    merged_df = merged_df.dropna(subset=['Adj Close'])
    merged_df['date'] = merged_df['date'].astype(str)
    merged_df['date'] = merged_df['date'].str.replace('-', '/')
    merged_df.fillna(0, inplace=True)
    
    return merged_df



### AAPL

file_news = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/AAPL_sentiments.csv"
file_twitter = '/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_AAPL_sentiments.csv'
file_stocks = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/stocks/AAPL.csv"

df_sentiments = merge_sentiments(file_news, file_twitter, file_stocks)
df_sentiments.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/Finalised_datasets/source_price_AAPL_TH.csv", index=False)

### AMZN

file_news = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/AMZN_sentiments.csv"
file_twitter = '/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_AMZN_sentiments.csv'
file_stocks = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/stocks/AMZN.csv"

df_sentiments = merge_sentiments(file_news, file_twitter, file_stocks)
df_sentiments.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/Finalised_datasets/source_price_AMZN_TH.csv", index=False)

### MSFT

file_news = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/MSFT_sentiments.csv"
file_twitter = '/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_MSFT_sentiments.csv'
file_stocks = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/stocks/MSFT.csv"

df_sentiments = merge_sentiments(file_news, file_twitter, file_stocks)
df_sentiments.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/Finalised_datasets/source_price_MSFT_TH.csv", index=False)

### NVDA

file_news = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/NVDA_sentiments.csv"
file_twitter = '/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_NVDA_sentiments.csv'
file_stocks = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/stocks/NVDA.csv"

df_sentiments = merge_sentiments(file_news, file_twitter, file_stocks)
df_sentiments.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/Finalised_datasets/source_price_NVDA_TH.csv", index=False)



'''### AAPL

file1 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AAPL_sentiment.csv"
file2 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AAPL_sentiment.csv"
file3 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AAPL_sentiment.csv"

merged_df = merge_datasets(file1, file2, file3)
merged_df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/AAPL_sentiments.csv", index=False)

### AMZN

file1 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AMZN_sentiment.csv"
file2 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AMZN_sentiment.csv"
file3 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AMZN_sentiment.csv"

merged_df = merge_datasets(file1, file2, file3)
merged_df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/AMZN_sentiments.csv", index=False)

### MSFT

file1 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_MSFT_sentiment.csv"
file2 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_MSFT_sentiment.csv"
file3 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_MSFT_sentiment.csv"

merged_df = merge_datasets(file1, file2, file3)
merged_df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/MSFT_sentiments.csv", index=False)

### NVDA

file1 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_NVDA_sentiment.csv"
file2 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_NVDA_sentiment.csv"
file3 = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_NVDA_sentiment.csv"

merged_df = merge_datasets(file1, file2, file3)
merged_df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/NVDA_sentiments.csv", index=False)'''



'''#####################################
##########  Other dataset  ##########
#####################################

### AAPL

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AAPL_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AAPL_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### AMZN

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AMZN_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_AMZN_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### MSFT

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_MSFT_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_MSFT_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### NVDA

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_NVDA_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/other_NVDA_sentiment.csv", index=False)

print(f"Klart med {file_path}")

#####################################
########  ABC dataset  ########
#####################################

### AAPL

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AAPL_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AAPL_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### AMZN

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AMZN_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_AMZN_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### MSFT

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_MSFT_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_MSFT_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### NVDA

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_NVDA_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/abc_NVDA_sentiment.csv", index=False)

print(f"Klart med {file_path}")

#####################################
########  Headlines dataset  ########
#####################################

### AAPL

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AAPL_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AAPL_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### AMZN

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AMZN_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_AMZN_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### MSFT

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_MSFT_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_MSFT_sentiment.csv", index=False)

print(f"Klart med {file_path}")

### NVDA

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_NVDA_processed.csv"

df = pd.read_csv(file_path)

df['sentiment'] = df['Headline'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df = df.groupby('Date').agg({'compound': 'mean', 'TH': 'mean'}).reset_index()
df = df.rename(columns={'compound': 'mean_compound_reuters'})
df = df.rename(columns={'Date': 'date'})
df.columns = [['date', 'mean_compound', 'TH']]

df.to_csv("/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/new_dataset/processed_news/headlines_NVDA_sentiment.csv", index=False)

print(f"Klart med {file_path}")'''
