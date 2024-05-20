from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

analyzer = SentimentIntensityAnalyzer()

def get_tweet_timestamp(time):
    offset = 1288834974657
    tstamp = (time >> 22) + offset
    utcdttime = datetime.utcfromtimestamp(tstamp/1000)
    # print(str(time) + " : " + str(tstamp) + " => " + str(utcdttime))
    return utcdttime

def get_sentiment(text):
    return analyzer.polarity_scores(text)

file1_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/Company_Tweet.csv"
file2_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/Tweet.csv"
filter_date = '2017-12-22 00:00:00'

# Read the CSV files into DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)
df_merged = pd.merge(df2, df1[['tweet_id', 'ticker_symbol']], on='tweet_id', how='left') # Merge the DataFrames on the 'tweet_id' column
df_filtered = df_merged[df_merged['ticker_symbol'] == "AMZN"] # Filter out everything but Amazon-related tweets
df_filtered['datetime'] = df_filtered['tweet_id'].apply(get_tweet_timestamp) # Create a new 'datetime' column from tweet id's
df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])
df = df_filtered[df_filtered['datetime'] >= filter_date]

df['sentiment'] = df['body'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['datetime'] = df['datetime'].dt.date
df = df.groupby('datetime')['compound'].mean().reset_index()
df = df.rename(columns={'compound': 'mean_compound_twitter'})
df = df.rename(columns={'datetime': 'date'})
df.columns = [['date', 'mean_compound_twitter']]

df.columns = ['_'.join(col).strip() for col in df.columns.values]
df['date'] = pd.to_datetime(df['date'])

df.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_sentiment_2017-2020.csv', index=False)
