'''
Module Name: text_preprocessing.py
Author: Dylan M. Crain

Description: Used to clean up the raw news title data gathered.
             Calculates the VADER score (sentiment measure) for
             each title and averages them over a given day.

             Furthermore, when no titles are available, the
             value 0 is given -- which means neutral sentiment.
'''


# ====================
# Import Libraries
# ====================


from os.path import join
import datetime
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ====================
# Support Defs
# ====================


def reformat_date(date):
    '''
    Function
    --------
    reformat_date():
        Format date object into Month/Day/Year.

    Parameters
    ----------
    date: datetime object
        Date to be transformed into common string.

    Returns
    -------
    formatted: str
        String form of the date object of form specified.
    '''
    fmt = '%m/%d/%Y'
    formatted = date.strftime(fmt)

    return formatted


def is_empty(line):
    '''
    Function
    --------
    is_empty():
        Checks to see if a read-in line is empty.
        Returns True, if so.

    Parameters
    ----------
    line: str
        Line from the raw file.

    Returns
    -------
    : bool
        True if the line is empty.
    '''
    if not line.strip():
        return True
    else:
        return False


def is_weekend(date):
    '''
    Function
    --------
    is_weekend():
        Checks to see if the date provided is a weekday.
        If so, returns True.

    Parameters
    ----------
    date: datetime object
        Date to be transformed into common string.

    Returns
    -------
    : bool
        True if the date is during the weekend
    '''
    # Definition of the weekend parameters
    weekend = ['Sat', 'Sun']

    if date.strftime('%a') in weekend:
        return True
    else:
        return False


def cleanup_raw_file(file_path, save_path, delim=','):
    '''
    Function
    --------
    cleanup_raw_file():
        Takes the raw file and puts the dates into proper format
        (Month/Day/Year) with the news title next to it in a delimited file.

    Parameters
    ----------
    file_path: str
        Path to the raw data file.
    save_path: str
        Path to save the cleaned file to.
    delim: str
        File delimiter for this intermediary cleaned file.
        Defaults to ",".

    Returns
    -------
    cleaned_data: ndarray
        Array of two columns and data point # of rows.
        The first is the date, and the second is the news title.
    '''
    # Format found to be turned into datetime and initialize.
    fmt = '%b %d, %Y'
    cleaned_data = []
    item_data = []

    # Open the raw data file.
    with open(file_path, 'r') as fid:
        # For each line in the file.
        for line in fid:
            line = line.strip()
            if not is_empty(line):
                # Try to transform a date.
                try:
                    date = datetime.datetime.strptime(line, fmt)
                # If failure, then the line is the news title.
                except ValueError:
                    if not is_weekend(date):
                        item_data.append(line.lower())
                # If date was successful, format it and add to new data.
                else:
                    if not is_weekend(date):
                        formatted_data = reformat_date(date)
                        item_data.append(formatted_data)

            # When item_data is filled, add to cleaned_data.
            if len(item_data) == 2:
                cleaned_data.append(item_data)
                item_data = []

    # Flip data to be older to newer dates, save, and return.
    cleaned_data = np.flipud(cleaned_data)
    np.savetxt(save_path, cleaned_data, fmt=('%s' + delim + '%s'))
    return cleaned_data


def get_vader_score(sentence):
    '''
    Function
    --------
    get_vader_score():
        For a given sentence, gets the VADER score, i.e., a sentiment score of
        how positive or negative a sentence is taken to be.

    Parameters
    ----------
    sentence: str
        Sentence to derive the sentiment from.

    Returns
    -------
    score: str
        String representation of a float from -1 to 1.
        Where -1 is fully negative and 1 is fully positive.
    '''
    # Begin Sentiment object and calculate scores.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # Get compund score and return.
    score = '%.4f' % sentiment_dict['compound']
    return score


def aggregate_vader_scores(cleaned_data, save_path):
    '''
    Function
    --------
    aggregate_vader_scores():
        Gets the vader scores for each title example to further clean the news
        (textual) data used in the predictions.

    Parameters
    ----------
    cleaned_data: ndarray
        Array of two columns and data point # of rows.
        The first is the date, and the second is the news title.
    save_path: str
        Path to save the cleaned file to.

    Returns
    -------
    scored_data: ndarray
        Similar data structure to cleaned_data.
        The exception is that instead of news titles
        being applied to each example, the VADER score is there instead.
    '''
    # Initialize the data structure.
    scored_data = np.zeros_like(cleaned_data)
    scored_data[:, 0] = cleaned_data[:, 0]

    # For each example, find the VADER/sentimate score.
    for index, sentence in enumerate(cleaned_data[:, 1]):
        score = get_vader_score(sentence)
        scored_data[index, 1] = score

    # Save and return the data.
    np.savetxt(save_path, scored_data, fmt=('%s,%s'))
    return scored_data


def get_date_range(start_date, days):
    '''
    Function
    --------
    get_date_range():
        Get the dates from a start and after a set of days (excluding
        weekends).

    Parameters
    ----------
    start_data: datetime object
        Date to begin at for the range.
    days: int
        The number of days beyond the start date to get values for.

    Returns
    -------
    date_range: list
        List of strings for the date range requested.
    '''
    # Get range of datetime objects.
    date_objects = [start_date - datetime.timedelta(days=idx)
                    for idx in range(days)]
    # Remove weekends.
    weekdays = [data for data in date_objects if not is_weekend(data)]
    # Format the dates to appropriate, default values.
    date_range = [reformat_date(date) for date in weekdays]

    return date_range


def get_daily_vader_average(scored_data, start_date, days, save_path):
    '''
    Function
    --------
    get_daily_vader_average():
        Takes the data for each example with the VADER scores and averages
        these values across each date. For dates with no news articles, a value
        of zero is assigned, which is a neutral sentiment.

    Parameters
    ----------
    scored_data: ndarray
        The exception is that instead of news titles
        being applied to each example, the VADER score is there instead.
    start_date: datetime object
        Date to begin at for the range.
    days: int
        The number of days beyond the start date to get values for.
    save_path: str
        Path to save the cleaned file to.

    Returns
    -------
    : None
        Simply saves the final cleanup file.
    '''
    # Get appropriate range of dates (all) and get scores.
    date_range = get_date_range(start_date, days)
    scores = scored_data[:, 1].astype(float)

    # Initialize the final data structure.
    cleaned_text_data = np.zeros((len(date_range), 2)).astype(str)
    cleaned_text_data[:, 0] = date_range[::-1]

    # Iterate over the unique dates in the scored data and avg their vals.
    unique_dates = np.unique(scored_data[:, 0])
    for date in unique_dates:
        # Find scores for a given date.
        date_loc = np.argwhere(scored_data[:, 0] == date)
        selected_scores = scores[date_loc]

        # Average the scores over the day and return as string.
        average_score = np.average(selected_scores)
        average_score = '%.4f' % average_score

        # Assign the averaged score to its appropriate date in the final data.
        new_loc = np.argwhere(cleaned_text_data[:, 0] == date)
        cleaned_text_data[new_loc, 1] = average_score

    # Saves result to a file for later retrieval.
    np.savetxt(save_path, cleaned_text_data, fmt=('%s,%s'))


# ====================
# Main Def
# ====================


def main(raw_path, saves, start_date, days, calc_scores=True):
    '''
    Function
    --------
    main():
        Runs all of the aforementioned steps to pre-process the text data.

    Parameters
    ----------
    raw_path: str
        Path to the raw textual data.
    saves: tuple
        Tuple of file paths to save the intermediary and final, scored, data
        set.
    start_date: datetime object
        Date to begin at for the range.
    days: int
        The number of days beyond the start date to get values for.
    calc_scores: bool
        If true, calculate all the VADER scores; otherwise, just load the old
        file.

    Returns
    -------
    : None
        Saves the intermediary and final cleanup files.
    '''
    # Variable initialization.
    formatted, scored, final = saves
    delim = '@'

    # Perform statements above.
    cleaned_data = cleanup_raw_file(raw_path, formatted, delim)
    if calc_scores:
        scored_data = aggregate_vader_scores(cleaned_data, scored)
    else:
        scored_data = np.genfromtxt(scored, delimiter=',', dtype=str)
    get_daily_vader_average(scored_data, start_date, days, final)


# ====================
# Run Entire Module
# ====================


if __name__ == '__main__':
    # --------------------
    # Variable initialization
    # --------------------
    # Conversion factor and shift.
    DAYS_IN_YEAR = 365
    DATE_SHIFT = 2

    # Get the start date and days to move.
    START_DATE = datetime.datetime(2021, 11, 24)
    YEARS = 4
    DAYS = YEARS * DAYS_IN_YEAR + DATE_SHIFT

    # Data paths.
    DATA_PATH = join('..', 'data', 'textual')
    RAW_PATH = join(DATA_PATH, 'raw_data.txt')

    # Save paths.
    SAVE_REFORMAT = join(DATA_PATH, 'cleaned_data_reformatted.csv')
    SAVE_ALL_SCORES = join(DATA_PATH, 'cleaned_data_all_scores.csv')
    SAVE_CLEANED = join(DATA_PATH, 'cleaned_data.csv')

    # --------------------
    # Run Main
    # --------------------
    main(RAW_PATH, (SAVE_REFORMAT, SAVE_ALL_SCORES, SAVE_CLEANED),
         START_DATE, DAYS)
