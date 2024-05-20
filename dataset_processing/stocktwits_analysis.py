from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

file_path = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Stocktwits/AMZN_pre_processed.csv"
df = pd.read_csv(file_path)

print(df.head())

########################
### Filter out dates ###
########################
filter_date = '2020-01-01'

# Convert the 'created_at' column to datetime format
df['created_at'] = pd.to_datetime(df['created_at'])

# Filter the DataFrame to keep only rows with 'created_at' after the filter_date
df_filtered = df[df['created_at'] <= filter_date]

# Display the first few rows of the filtered DataFrame
print(df_filtered.shape)

# Select only the relevant columns
relevant_columns = ['created_at', 'body', 'sentiment', 'stock']
df = df_filtered[relevant_columns]

# Display the first few rows of the final DataFrame
print(df.head())


analyzer = SentimentIntensityAnalyzer()

# 3. Function to get sentiment scores
def get_sentiment(text):
    return analyzer.polarity_scores(text)

df['sentiment'] = df['body'].apply(lambda x: get_sentiment(x))
df['compound'] = df['sentiment'].apply(lambda x: x['compound'])
df['created_at'] = df['created_at'].dt.date
df = df.groupby('created_at')['compound'].mean().reset_index()
df = df.rename(columns={'compound': 'mean_compound_stocktwits'})
df = df.rename(columns={'created_at': 'date'})
df.columns = [['date', 'mean_compound_stocktwits']]

df.columns = ['_'.join(col).strip() for col in df.columns.values]
df['date'] = pd.to_datetime(df['date'])

df.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Stocktwits/stocktwits_sentiment.csv', index=False)

# Print the first few rows to verify
print(df.head())