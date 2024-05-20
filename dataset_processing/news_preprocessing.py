from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
import re

keywords = ["Amazon", "AMZN", "Jeff Bezos", "Bezos", "Andy Jassy", "Amazon Web Services", "AWS"]

def contains_keywords(headline, keywords):
    # Create a regex pattern for each keyword to match whole words
    patterns = [re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) for keyword in keywords]
    # Check if any pattern matches the headline
    return any(pattern.search(headline) for pattern in patterns)

start_date = '2017-12-22'
end_date = '2020-01-01'

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    return analyzer.polarity_scores(text)

#####################################
###########  Reuters  ###############
#####################################

file_path_reuters = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/reuters_headlines.csv"  # Update this to the path of your CSV file
df_reuters = pd.read_csv(file_path_reuters)

df_reuters = df_reuters[df_reuters['Headlines'].apply(lambda x: contains_keywords(x, keywords))]
df_reuters['Time'] = pd.to_datetime(df_reuters['Time'], format='%b %d %Y')
# df_reuters = df_reuters[~((df_reuters['Time'] <= start_date) & (df_reuters['Time'] >= end_date))]
df_reuters = df_reuters[~((df_reuters['Time'] >= end_date))]
df_reuters = df_reuters[~((df_reuters['Time'] <= start_date))]

df_reuters.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/reuters_headlines_preprocessed.csv', index=False)

df_reuters['sentiment'] = df_reuters['Headlines'].apply(lambda x: get_sentiment(x))
df_reuters['compound'] = df_reuters['sentiment'].apply(lambda x: x['compound'])
df_reuters = df_reuters.groupby('Time')['compound'].mean().reset_index()
df_reuters = df_reuters.rename(columns={'compound': 'mean_compound_reuters'})
df_reuters = df_reuters.rename(columns={'Time': 'date'})
df_reuters.columns = [['date', 'mean_compound_reuters']]

df_reuters.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/reuters_sentiments.csv', index=False)


#####################################
###########  Guardian  ##############
#####################################

def parse_date(date_str):
    try:
        # Try parsing date with day-month-year format
        return pd.to_datetime(date_str, format='%d-%b-%y')
    except ValueError:
        # If it fails, try month-year format, assuming the first day of the month
        return pd.to_datetime(date_str, format='%b-%y')


file_path_guardian = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/guardian_headlines.csv"  # Update this to the path of your CSV file
df_guardian = pd.read_csv(file_path_guardian)

df_guardian['Time'] = df_guardian['Time'].apply(parse_date)
df_guardian = df_guardian[df_guardian['Headlines'].apply(lambda x: contains_keywords(x, keywords))]
df_guardian = df_guardian[~((df_guardian['Time'] >= end_date))]
df_guardian = df_guardian[~((df_guardian['Time'] <= start_date))]

df_guardian.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/guardian_headlines_preprocessed.csv', index=False)

df_guardian['sentiment'] = df_guardian['Headlines'].apply(lambda x: get_sentiment(x))
df_guardian['compound'] = df_guardian['sentiment'].apply(lambda x: x['compound'])
df_guardian = df_guardian.groupby('Time')['compound'].mean().reset_index()
df_guardian = df_guardian.rename(columns={'compound': 'mean_compound_guardian'})
df_guardian = df_guardian.rename(columns={'Time': 'date'})
df_guardian.columns = [['date', 'mean_compound_guardian']]

df_guardian.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/guardian_sentiments.csv', index=False)


#####################################
#############  CNBC  ################
#####################################

def parse_datetime(date_str):
    # Remove 'ET' from the date string
    date_str = date_str.replace('ET', '').strip()
    # Define the date format
    date_format = '%I:%M %p %a, %d %B %Y'
    # Parse the date string to datetime
    return pd.to_datetime(date_str, format='mixed')

start_date_dt = pd.to_datetime(start_date).date()
end_date_dt = pd.to_datetime(end_date).date()

file_path_cnbc = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_headlines.csv"  # Update this to the path of your CSV file
df_cnbc = pd.read_csv(file_path_cnbc)

df_cnbc = df_cnbc.dropna(subset=['Time'])
df_cnbc['Time'] = df_cnbc['Time'].apply(parse_datetime)
df_cnbc['Time'] = df_cnbc['Time'].dt.date
df_cnbc = df_cnbc[df_cnbc['Headlines'].apply(lambda x: contains_keywords(x, keywords))]
df_cnbc = df_cnbc[~((df_cnbc['Time'] >= end_date_dt))]
df_cnbc = df_cnbc[~((df_cnbc['Time'] <= start_date_dt))]

df_cnbc.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_headlines_preprocessed.csv', index=False)

df_cnbc['sentiment'] = df_cnbc['Headlines'].apply(lambda x: get_sentiment(x))
df_cnbc['compound'] = df_cnbc['sentiment'].apply(lambda x: x['compound'])
df_cnbc = df_cnbc.groupby('Time')['compound'].mean().reset_index()
df_cnbc = df_cnbc.rename(columns={'compound': 'mean_compound_cnbc'})
df_cnbc = df_cnbc.rename(columns={'Time': 'date'})
df_cnbc.columns = [['date', 'mean_compound_cnbc']]

df_cnbc.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_sentiments.csv', index=False)



#####################################
############  Other  ################ (source: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv)
#####################################

file_path_other = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/raw_partner_headlines.csv"  # Update this to the path of your CSV file
df_other = pd.read_csv(file_path_other)

print(df_other.head())

df_other['date'] = pd.to_datetime(df_other['date'])
df_other['date'] = df_other['date'].dt.date
df_other = df_other[~((df_other['date'] >= end_date_dt))]
df_other = df_other[~((df_other['date'] <= start_date_dt))]
df_other = df_other[df_other['headline'].apply(lambda x: contains_keywords(x, keywords))]

df_other = df_other[['headline', 'date', 'stock']]

print(df_other.head())

df_other.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/raw_partner_headlines_preprocessed.csv', index=False)

df_other['sentiment'] = df_other['headline'].apply(lambda x: get_sentiment(x))
df_other['compound'] = df_other['sentiment'].apply(lambda x: x['compound'])
df_other = df_other.groupby('date')['compound'].mean().reset_index()
df_other = df_other.rename(columns={'compound': 'mean_compound_other'})
df_other = df_other.rename(columns={'date': 'date'})
df_other.columns = [['date', 'mean_compound_other']]

df_other.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/other_sentiments.csv', index=False)

df_reuters.columns = ['_'.join(col).strip() for col in df_reuters.columns.values]
df_guardian.columns = ['_'.join(col).strip() for col in df_guardian.columns.values]
df_cnbc.columns = ['_'.join(col).strip() for col in df_cnbc.columns.values]
df_other.columns = ['_'.join(col).strip() for col in df_other.columns.values]

df_reuters['date'] = pd.to_datetime(df_reuters['date'])
df_guardian['date'] = pd.to_datetime(df_guardian['date'])
df_cnbc['date'] = pd.to_datetime(df_cnbc['date'])
df_other['date'] = pd.to_datetime(df_other['date'])


# Merge the dataframes on the 'date' column
merged_df = pd.merge(df_reuters, df_guardian, on='date', how='outer', suffixes=('_df_reuters', 'df_guardian'))
merged_df = pd.merge(merged_df, df_cnbc, on='date', how='outer', suffixes=('', 'df_cnbc'))
merged_df = pd.merge(merged_df, df_other, on='date', how='outer', suffixes=('', 'df_other'))

# Sort the DataFrame by the 'date' column
merged_df = merged_df.sort_values('date')

merged_df.fillna(0, inplace=True)

merged_df.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/news_sentiments_2017-2020.csv', index=False)