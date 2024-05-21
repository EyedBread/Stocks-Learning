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
    '''
    raw_data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    raw_data = raw_data[:, 1:].astype('float32')

    return raw_data


def normalize_data(raw_data):
    '''
    Function
    --------
    normalize_data():
        Normalizes each feature of the data between 0 and 1.

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

    Returns
    -------
    raw_data: ndarray
        Same as parameter, but with each feature normalized between 0 & 1.
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    raw_data = scaler.fit_transform(raw_data)

    return raw_data


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


# ====================
# Main Def
# ====================


def data_prep(data_path, horizon, days_forward, end_split):
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
        Path to the cleaned, combined data set.
    horizon: int
        Number of previous days used to predict per time-step.
    days_forward: int
        Number of days forward to predict the movement of stock.
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
    # Get raw data for 2D structure and get dimensions.
    raw_data = normalize_data(load_raw_data(data_path))
    raw_sample_size, feature_size = raw_data.shape

    # Get corrected sample size and the closing prices.
    sample_size = get_sample_size(raw_sample_size, horizon, days_forward)
    closing_prices = raw_data[:, 0]

    # Create the sequences of X and the stock closing day movements.
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
    DATA_PATH = join('..', 'data', 'combined', 'cleaned_data.csv')

    HORIZON = 10
    DAYS_FORWARD = 1
    END_SPLIT = 40

    # --------------------
    # Main functionality
    # --------------------
    split_y, split_X = data_prep(DATA_PATH, HORIZON, DAYS_FORWARD, END_SPLIT)
