from openai import OpenAI
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
import re

keywords = ["Amazon", "AMZN", "Jeff Bezos", "Bezos", "Andy Jassy", "Amazon Web Services", "AWS"]

start_date = '2017-01-01'
end_date = '2020-01-01'
start_date_dt = pd.to_datetime(start_date).date()
end_date_dt = pd.to_datetime(end_date).date()

analyzer = SentimentIntensityAnalyzer()
print(os.environ.get("OPENAI_API_KEY")) #key should now be available
client = OpenAI(api_key="sk-proj-xQXwBSxseQziVSLApBpQT3BlbkFJmcL45UWX08omWpLDhR76")

def contains_keywords(headline, keywords):
    # Create a regex pattern for each keyword to match whole words
    patterns = [re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE) for keyword in keywords]
    # Check if any pattern matches the headline
    return any(pattern.search(headline) for pattern in patterns)

def parse_datetime(date_str):
    # Remove 'ET' from the date string
    date_str = date_str.replace('ET', '').strip()
    # Define the date format
    date_format = '%I:%M %p %a, %d %B %Y'
    # Parse the date string to datetime
    return pd.to_datetime(date_str, format='mixed')

def get_sentiment(text):
    return analyzer.polarity_scores(text)



class RelevanceEstimator:
    def __init__(self):
        self.counter = 0

    def get_relevance2(self, headline, size):
        if self.counter % 20 == 0:
            print(f"Progress is {round(self.counter/size, 2)}. On row {self.counter}")
        self.counter += 1
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst. We have news headlines about various big companies and need to estimate how long each will stay relevant (1-60 days). Consider the nature of the news (e.g., earnings reports, product launches, mergers, controversies) and the typical news cycle for similar events. Provide only the estimated relevance duration in days as a single integer and nothing more."},
                {"role": "user", "content": headline}
            ]
        )

        if self.counter == size:
            self.counter = 0
        #relevance_str = response['choices'][0]['message']['content'].strip()
        relevance_str = response.choices[0].message.content.strip()
        try:
            relevance = int(relevance_str)
        except ValueError:
            relevance = None  # handle cases where the response is not a valid integer
        return relevance
    
    def get_relevance(self, headlines, size):
        if self.counter % 100 == 0:
            print(f"Progress is {round(self.counter/size, 2)}. On row {self.counter}")
        self.counter += 50 # Batch size of 50

        # Create a single message for all headlines
        headlines_text = "\n".join([f"{i+1}. {headline}" for i, headline in enumerate(headlines)])
        system_message = (
            "You are a data analyst. We have news headlines about various big companies and need to estimate how long "
            f"each will stay relevant (1-60 days). Consider the nature of the news (e.g., earnings reports, product "
            "launches, mergers, controversies), the typical news cycle for similar events, and the number of similar "
            f"headlines ({size}). Provide the estimated relevance duration in days as a single integer for each headline, "
            "numbered sequentially, in the same order. Sample format: '1: 15, 2: 30, ...'.\n\n, all the way up to 50. Do NOT reply in any other format, and don't finish with a '.'"
            f"Headlines:\n{headlines_text}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": headlines_text}
            ]
        )

        if self.counter >= size:
            self.counter = 0
        
        relevance_text = response.choices[0].message.content.strip()
        
        # Parse the response
        relevance_map = {}
        # print(relevance_text)
        for line in relevance_text.split(","):
            number, days = line.split(":")
            relevance_map[int(number.strip()) - 1] = int(days.strip())
        
        # Return the relevance in the same order as the headlines
        return [relevance_map[i] for i in range(len(headlines))]


'''completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are analysing data, estimating how many days news articles relating to Amazon's stock hold relevance. Your replies should only be a number and nothing more."},
    {"role": "user", "content": "The charts show Amazon's stock is bottoming, primed for a 20percent gain, Jim Cramer says."}
  ]
)
print(completion.choices[0].message)'''

if __name__ == "__main__":
  file_path_cnbc = "/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_headlines.csv"  # Update this to the path of your CSV file
  '''df_cnbc = pd.read_csv(file_path_cnbc)

  df_cnbc = df_cnbc.dropna(subset=['Time'])
  df_cnbc['Time'] = df_cnbc['Time'].apply(parse_datetime)
  df_cnbc['Time'] = df_cnbc['Time'].dt.date
  df_cnbc = df_cnbc[df_cnbc['Headlines'].apply(lambda x: contains_keywords(x, keywords))]
  df_cnbc = df_cnbc[~((df_cnbc['Time'] >= end_date_dt))]
  df_cnbc = df_cnbc[~((df_cnbc['Time'] <= start_date_dt))]
  df_cnbc['relevance'] = df_cnbc['Headlines'].apply(get_relevance)

  df_cnbc.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_headlines_preprocessed.csv', index=False)

  print("kör")

  df_cnbc['sentiment'] = df_cnbc['Headlines'].apply(lambda x: get_sentiment(x))
  df_cnbc['compound'] = df_cnbc['sentiment'].apply(lambda x: x['compound'])
  # df_cnbc = df_cnbc.groupby('Time')['compound'].mean().reset_index()
  df_cnbc = df_cnbc.groupby('Time').agg({'compound': 'mean', 'relevance': 'mean'}).reset_index()
  df_cnbc = df_cnbc.rename(columns={'compound': 'mean_compound_cnbc'})
  df_cnbc = df_cnbc.rename(columns={'Time': 'date'})
  #df_cnbc.columns = [['date', 'mean_compound_cnbc', 'relevance']]
  #df_cnbc = df_cnbc[['date', 'mean_compound_cnbc', 'relevance']]
  df_cnbc.columns = [['date', 'mean_compound_reuters', 'relevance']]

  df_cnbc.to_csv('/Users/marku/Documents/Plugg/DD2424 - Deep Learning/Stocks-Learning/data/original_dataset/News/cnbc_sentiments.csv', index=False)'''

  
  print(get_relevance("Skyworks growing with Apple"))
  print(get_relevance("Jabil Circuit: Stock To Benefit From Strong iPhone Sales"))
  print(get_relevance("Stocks Reverse Lower; Apple Plunges Below Support"))
  print(get_relevance("Apple's Beats Electronics Sued by Monster LLC - Analyst Blog"))
