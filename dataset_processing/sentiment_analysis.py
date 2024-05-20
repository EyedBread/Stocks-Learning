from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
#note: depending on how you installed (e.g., using source code download versus pip install), you may need to import like this:
#from vaderSentiment import SentimentIntensityAnalyzer

'''# --- examples -------
sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
             "VADER is smart, handsome, and funny!",  # punctuation emphasis handled correctly (sentiment intensity adjusted)
             "VADER is very smart, handsome, and funny.", # booster words handled correctly (sentiment intensity adjusted)
             "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
             "VADER is VERY SMART, handsome, and FUNNY!!!", # combination of signals - VADER appropriately adjusts intensity
             "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", # booster words & punctuation make this close to ceiling for score
             "VADER is not smart, handsome, nor funny.",  # negation sentence example
             "The book was good.",  # positive sentence
             "At least it isn't a horrible book.",  # negated negative sentence with contraction
             "The book was only kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
             "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
             "Today SUX!",  # negative slang with capitalization emphasis
             "Today only kinda sux! But I'll get by, lol", # mixed sentiment example with slang and constrastive conjunction "but"
             "Make sure you :) or :D today!",  # emoticons handled
             "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # emojis handled
             "Not bad at all"  # Capitalized negation
             ]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))
'''

start_date = '2019-07-23'
end_date = '2020-01-01'
start_date_dt = pd.to_datetime(start_date).date()
end_date_dt = pd.to_datetime(end_date).date()

file_path_news = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/news_sentiments.csv"
file_path_stocktwits = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Stocktwits/stocktwits_sentiment.csv"
file_path_twitter = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/Tweets/twitter_sentiment.csv"

data_news = pd.read_csv(file_path_news)
data_stocktwits = pd.read_csv(file_path_stocktwits)
data_twitter = pd.read_csv(file_path_twitter)

merged_df = pd.merge(data_news, data_stocktwits, on='date', how='outer')
merged_df = pd.merge(merged_df, data_twitter, on='date', how='outer')

merged_df['date'] = pd.to_datetime(merged_df['date'])
merged_df['date'] = merged_df['date'].dt.date
merged_df = merged_df.sort_values('date')
merged_df.fillna(0, inplace=True)

merged_df.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/sentiments.csv', index=False)

file_path_stocks = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/stockdata/Amazon/AMZN.csv"
data_amzn = pd.read_csv(file_path_stocks)

data_amzn = data_amzn.rename(columns={'Date': 'date'})
data_amzn['date'] = pd.to_datetime(data_amzn['date'])
data_amzn['date'] = data_amzn['date'].dt.date
data_amzn = data_amzn[~((data_amzn['date'] >= end_date_dt))]
data_amzn = data_amzn[~((data_amzn['date'] <= start_date_dt))]
data_amzn = data_amzn[['date', 'Adjusted Close']]
data_amzn = data_amzn.rename(columns={'Adjusted Close': 'Adj Close'})

merged_df = pd.merge(merged_df, data_amzn, on='date', how='outer', suffixes=('_df_merged', 'df_amzn'))
merged_df = merged_df.sort_values('date')

merged_df = merged_df.dropna(subset=['Adj Close'])

merged_df['date'] = merged_df['date'].astype(str)

merged_df['date'] = merged_df['date'].str.replace('-', '/')
#merged_df['date'] = merged_df['date'].dt.strftime('%Y/%m/%d')

merged_df.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/amzn_source_price.csv', index=False)