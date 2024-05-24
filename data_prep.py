'''
Module Name: data_prep.py
Author: Dylan M. Crain

Description: Turns cleaned data into proper sequences, normalizes, and splits
             it into train, validate, and test portions.
'''


# ====================
# Import Libraries
# ====================


from os.path import join
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# ====================
# Supporting Defs
# ====================


def load_raw_data(data_path):
    '''
    Function
    --------
    load_raw_data():
        Loads the cleaned-up, combined data without the date column.

    Parameters
    ----------
    data_path: str
        Path to the cleaned, combined data set.

    Returns
    -------
    raw_data: ndarray
        2D array of the data with each row being a date and the columns being:

        0: Closing Price
        1: Volume
        2: Open
        3: High
        4: Low
        5: VADER Score.
        ...
        Last column: relevance score
    '''
    raw_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    raw_data = raw_data[:, 1:].astype('float32') #Drop the date column

    return raw_data


def normalize_data(raw_data, use_relevance_scores=False):
    '''
    Function
    --------
    normalize_data():
        Normalizes each feature of the data between 0 and 1, except for relevance score.

    Parameters
    ----------
    raw_data: ndarray
        2D array of the data with each row being a date and the columns being:

        0: Closing Price
        1: Volume
        2: Open
        3: High
        4: Low
        5: VADER Score.
        ...
        Last column: relevance score
    use_relevance_scores: bool
        Whether the data contains relevance scores column, we don't normalize it.

    Returns
    -------
    raw_data: ndarray
        Same as parameter, but with each feature normalized between 0 & 1.
    '''

    data_to_normalize = raw_data


    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data_to_normalize)



    return normalized_data


def get_sample_size(raw_sample_size, horizon, days_forward):
    '''
    Function
    --------
    get_sample_size():
        Gets the size of the new sample, based on the recurrent conditions.

    Parameters
    ----------
    raw_sample_size: int
        Number of samples (dates) of the original dataset.
    horizon: int
        Number of previous days used to predict per time-step.
    days_forward: int
        Number of days forward to predict the movement of stock.

    Returns
    -------
    sample_size: int
        Number of samples of sequences considering the other parameters.
    '''
    sample_size = raw_sample_size - horizon - days_forward + 1
    return sample_size


def get_movements_y(horizon, days_forward, closing_prices):
    '''
    Function
    --------
    get_movements_y():
        Gets the movements of the data through time, i.e., 1 means the stock
        has moved up or stayed steady at from the last time-step. 0 means that
        the stock has moved down from the last time-step.

    Parameters
    ----------
    horizon: int
        Number of previous days used to predict per time-step.
    days_forward: int
        Number of days forward to predict the movement of stock.
    closing_prices: ndarray
        Prices at the end of the day throughout the day.

    Returns
    -------
    movements_y: ndarray
        1D vector of movements from observation day to prediction day. 1 for no
        change or increase in closing price. 0 for decrease in closin price.
    '''
    # Get prices at end of horizon (early) and at observation (later).
    early_prices = closing_prices[horizon - 1: -days_forward]
    late_prices = closing_prices[horizon + days_forward - 1:]

    # Price difference and apply boolean changes.
    price_diff = late_prices - early_prices
    movements_y = (price_diff >= 0).astype('float32')

    return movements_y


def get_sequence_X(raw_data, sample_size, horizon, feature_size):
    '''
    Function
    --------
    get_sequence_X():
        Get the different time sequences for each sample from the normalized
        data.

    Parameters
    ----------
    raw_data: ndarray
        2D array of the data with each row being a date and the columns being:

        0: Closing Price
        1: Volume
        2: Open
        3: High
        4: Low
        5: VADER Score.

        Normalized values.
    sample_size: int
        Number of samples of sequences considering the other parameters.
    horizon: int
        Number of previous days used to predict per time-step.
    feature_size: int
        Number of independent parameters at each time-step.

    Returns
    -------
    sequence_X: ndarray
        Input information for the deep learning model.
        3D array of dimensions (samples, time-steps, features).
   '''
    sequence_X = np.zeros((sample_size, horizon, feature_size))

    for index in range(sample_size):
        sequence_X[index, ...] = raw_data[index: index + horizon]

    return sequence_X


def split_datasets(movements_y, sequence_X, end_split):
    '''
    Function
    --------
    split_datasets():
        Splits the movements (y) and sequences (X) into train, validation, and
        test sets.

    Parameters
    ----------
    movements_y: ndarray
        1D vector of movements from observation day to prediction day. 1 for no
        change or increase in closing price. 0 for decrease in closin price.
    sequence_X: ndarray
        Input information for the deep learning model.
        3D array of dimensions (samples, time-steps, features).
    end_split: int
        Number of days to have in the valiation and test sets, e.g., if 40,
        then there are 40 samples in the validation and 40 samples in the test
        sets.

    Returns
    -------
    split_y: tuple
        (train_y, validate_y, test_y)
    split_X: tuple
        (train_X, validate_X, test_X)
    '''
    # Reshape the movements
    movements_y = np.reshape(movements_y, (1, len(movements_y)))

    # Split the movements
    train_y = movements_y[0, :2 * -end_split]
    validate_y = movements_y[0, 2 * -end_split: -end_split]
    test_y = movements_y[0, -end_split:]

    split_y = (train_y, validate_y, test_y)

    # Split the feature space
    train_X = sequence_X[:2 * -end_split, ...]
    validate_X = sequence_X[2 * -end_split: -end_split, ...]
    test_X = sequence_X[-end_split:, ...]

    split_X = (train_X, validate_X, test_X)

    # Return the split results
    return split_y, split_X


def get_dynamic_sequence_X(raw_data, relevance_scores, sample_size, max_horizon, feature_size, movements_y):
    '''
    Function
    --------
    get_dynamic_sequence_X():
        Get the time sequences for each sample from the normalized data with dynamic lengths
        based on relevance scores.

    Parameters
    ----------
    raw_data: ndarray
        2D array of the data with each row being a date and the columns being features.
    relevance_scores: ndarray
        1D array with relevance scores influencing additional days to the horizon.
    sample_size: int
        Number of samples of sequences considering the other parameters.
    horizon: int
        Number of previous days used to predict per time-step.
    feature_size: int
        Number of independent parameters at each time-step.

    Returns
    -------
    sequence_X: ndarray
        Input information for the deep learning model, with dynamic sequence lengths.
        3D array of dimensions (samples, dynamic time-steps, features).
    '''
    # Prepare a list to hold sequences since they will have varying lengths
    sequence_list = []
    shift_factor = len(raw_data) - len(movements_y) #Shifting factor
    if shift_factor < 1:
        raise ValueError('The number of relevance scores exceeds the number of data points.')
    if (len(raw_data) - len(relevance_scores) != 0):
        raise ValueError('The number of relevance scores does not match the number of data points.')
    print("shift_factor", shift_factor)
    print("max_horizon", max_horizon)

    for i in range(movements_y.shape[0]):
        predicted_value = movements_y[i]
        sequence = []
        for j in range(max(0, i - int(max_horizon) - 1), i + int(shift_factor)):
            element = raw_data[j]
            relevance_score = relevance_scores[j]
            if relevance_score + j >= shift_factor + i:
                sequence.append(element)
        
        while len(sequence) < max_horizon:
            # append -1 to indicate padding
            sequence.append([-1] * feature_size)
        sequence_list.append(sequence)

    return np.array(sequence_list, dtype='float32')

def drop_column(arr, idx):
    return np.delete(arr, idx, axis=1)

# ====================
# Main Def
# ====================



def data_prep(data_path, horizon, days_forward, end_split, use_relevance_scores=False):
    '''
    Function
    --------
    data_prep():
        Collects the raw data, normalizes it, splits it into sequences, and
        compartmentalizes it into a training, validation, and test sets. It
        does the same for the stock movements as well.

    Parameters
    ----------
    data_path: str
        Path to the cleaned, combined data set. NOTE: The last column should be relevance scores.
    horizon: int
        Number of previous days used to predict per time-step.
    days_forward: int
        Number of days forward to predict the movement of stock.
    end_split: int
        Number of days to have in the valiation and test sets, e.g., if 40,
        then there are 40 samples in the validation and 40 samples in the test
        sets.
    use_relevance_scores: bool
        Whether to use relevance scores to adjust the horizon.

    Returns
    -------
    split_y: tuple
        (train_y, validate_y, test_y)
    split_X: tuple
        (train_X, validate_X, test_X)
    '''
    # Load data into a pandas DataFrame
    raw_data = pd.read_csv(data_path)

    # Calculate SMAs
    raw_data['SMA_15'] = raw_data['Adj Close'].rolling(window=15).mean()
    raw_data['SMA_30'] = raw_data['Adj Close'].rolling(window=30).mean()
    raw_data['SMA_Indicator'] = np.where(raw_data['SMA_15'] > raw_data['SMA_30'], 1, 0)

    # Calculate Middle, Upper, and Lower Bollinger Bands
    period = 20
    raw_data['MBB'] = raw_data['Adj Close'].rolling(window=period).mean()
    std_dev = raw_data['Adj Close'].rolling(window=period).std()
    raw_data['UBB'] = raw_data['MBB'] + (2 * std_dev)
    raw_data['LBB'] = raw_data['MBB'] - (2 * std_dev)

    # Calculate Bollinger Indicator
    raw_data['Bollinger_Indicator'] = np.select(
        [
            raw_data['Adj Close'] > raw_data['UBB'],
            (raw_data['Adj Close'] <= raw_data['UBB']) & (raw_data['Adj Close'] > raw_data['MBB']),
            (raw_data['Adj Close'] < raw_data['MBB']) & (raw_data['Adj Close'] >= raw_data['LBB']),
            raw_data['Adj Close'] < raw_data['LBB']
        ],
        [1, 2, 3, 4]
    )

    # Calculate Close diff Upper Bollinger
    raw_data['Close_diff_UBB'] = raw_data['Adj Close'] - raw_data['UBB']
    # Calculate Close diff Lower Bollinger
    raw_data['Close_diff_LBB'] = raw_data['Adj Close'] - raw_data['LBB']

    
    # Drop rows with NaN values in the columns needed for analysis
    raw_data = raw_data.dropna(subset=['SMA_15', 'SMA_30', 'MBB', 'UBB', 'LBB'])
    print(raw_data.head())

    # Drop irrelevant columns based on their headers
    # 'Volume','Open','High','Low',
    columns_to_drop = ['date', 'Close', 'SMA_15', 'SMA_30', 'MBB', 'UBB', 'LBB',  'Close_diff_UBB', 'Close_diff_LBB', 'Bollinger_Indicator', 'SMA_Indicator']  # Add headers of columns to drop here , 'Volume','Open','High','Low'
    # columns_to_drop = ['Date'] #For the tesla dataset
    raw_data = raw_data.drop(columns=columns_to_drop)

    relevance_scores = None
    relevance_column = 'mean TH'  
    if use_relevance_scores:
        
        relevance_scores = raw_data[relevance_column].values
        relevance_scores = np.round(relevance_scores)
    # if column exists
    if relevance_column in raw_data.columns:
        raw_data = raw_data.drop(columns=[relevance_column])
    closing_prices = raw_data['Adj Close'].astype('float32').values #['Adj Close']
    raw_data = raw_data.astype('float32').values  # Convert DataFrame to numpy array


    raw_data = normalize_data(raw_data, use_relevance_scores)
    raw_sample_size, feature_size = raw_data.shape

    # Get corrected sample size and the closing prices.
    sample_size = get_sample_size(raw_sample_size, horizon, days_forward)
    print(sample_size)
    

    # Create the sequences of X and the stock closing day movements.
    movements_y = None
    sequence_X = None
    if use_relevance_scores:
        min_horizon = horizon
        # add horizon to all relevance scores
        relevance_scores = relevance_scores + min_horizon
        # Get the max relevance score to adjust the sample size.
        max_relevance_score = np.max(relevance_scores)
        print("max_relevance_score", max_relevance_score)
        sample_size = get_sample_size(raw_sample_size, min_horizon, days_forward)
        movements_y = get_movements_y(min_horizon, days_forward, closing_prices)
        sequence_X = get_dynamic_sequence_X(raw_data[:, :], relevance_scores, sample_size, max_relevance_score, feature_size, movements_y)

        # TODO : THIS IS FOR DEBUGGING, REMOVE LATER WHEN 100% SURE IT WORKS
        # print("y movement", movements_y.shape)

        # print(sequence_X.shape)
        # print(sequence_X[0])
        # print(movements_y[0])
        # print(sequence_X[1])
        # print(movements_y[1])
    else:
        movements_y = get_movements_y(horizon, days_forward, closing_prices)
        sequence_X = get_sequence_X(raw_data, sample_size, horizon, feature_size)
    # Split data sets into train, validate, and test.
    split_y, split_X = split_datasets(movements_y, sequence_X, end_split)

    return split_y, split_X


# ====================
# Running from Module
# ====================


if __name__ == '__main__':
    # --------------------
    # Variable Initialize
    # --------------------
    # Test dataset for variable length sequences
    DATA_PATH = join('.', 'stanford_project', 'data', 'combined', 'amzn_all_sources_WITH_TH_2017-2020.csv')

    HORIZON = 10
    DAYS_FORWARD = 1
    END_SPLIT = 10

    # --------------------
    # Main functionality
    # --------------------
    split_y, split_X = data_prep(DATA_PATH, HORIZON, DAYS_FORWARD, END_SPLIT, True)
