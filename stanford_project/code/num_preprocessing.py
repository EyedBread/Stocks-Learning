'''
Module Name: num_preprocessing.py
Author: Dylan M. Crain

Description: Cleans up the numerical data gathered.
             Removes the $ signs, limits the temporal
             scope to match that of the textual data,
             and inverts the time series.
'''


# ====================
# Import Libraries
# ====================


from os.path import join
import numpy as np


# ====================
# Supporting Defs
# ====================


def upload_raw_data(path):
    '''
    Function
    --------
    upload_raw_data():
        Uploads the raw data from the saved file.

    Parameters
    ----------
    path: str
        Path to the raw data file.

    Returns
    -------
    raw_data: ndarray
        2D array with the 1st column the dates and the following columns the
        stock market values for Tesla stock.
    '''
    raw_data = np.genfromtxt(path, delimiter=',', dtype=str)
    return raw_data


def reduce_data(raw_data, oldest_time):
    '''
    Function
    --------
    reduce_data():
        Reduces the data to a given later date.

    Parameters
    ----------
    raw_data: ndarray
        2D array with the 1st column the dates and the following columns the
        stock market values for Tesla stock.
    oldest_time: str
        Oldest allowed time in the dataset. Of form 'month/day/year'.

    Returns
    -------
    reduced_data: ndarray
        Data structure with fewer dates/examples as specified.
    '''
    loc = np.argwhere(raw_data[:, 0] == oldest_time)[0][0]
    reduced_data = raw_data[:(loc + 1), :]

    return reduced_data


def remove_dollar_signs(reduced_data):
    '''
    Function
    --------
    remove_dollar_sign():
        Removes the dollar signs from the stock information.

    Parameters
    ----------
    reduced_data: ndarray
        Data structure with fewer dates/examples as specified.

    Returns
    -------
    no_sign_data: ndarray
        Data structure with dollar signs removed.
    '''
    no_sign_data = np.char.replace(reduced_data, '$', '')
    return no_sign_data


def reverse_dates(no_sign_data):
    '''
    Function
    --------
    reverse_dates():
        Reverse the dates to be oldest to newest.

    Parameters
    ----------
    no_sign_data: ndarray
        Data structure with dollar signs removed.

    Returns
    -------
    reverse_data: ndarray
        Data structure with times reversed.
    '''
    reverse_data = np.zeros_like(no_sign_data)

    reverse_data[0, :] = no_sign_data[0, :]
    reverse_data[1:, :] = np.flipud(no_sign_data[1:, :])

    return reverse_data


# ====================
# Main Def
# ====================


def main(load_path, save_path, oldest_time):
    '''
    Function
    --------
    main():
        Takes the raw numerical (stock) data and reformats it to get the proper
        ordering of dates, eliminate dollar signs, and reduce the time-series
        to incorporate the text data.

    Parameters
    ----------
    load_path: str
        Path to the raw data file.
    save_path: str
        Path to save the cleaned data to.
    oldest_time: str
        Oldest allowed time in the dataset. Of form 'month/day/year'.

    Returns
    -------
    : None
        Saves the cleaned data file.
    '''
    # Upload raw data
    raw_data = upload_raw_data(load_path)

    # Perform transformations on said data.
    reduced_data = reduce_data(raw_data, oldest_time)
    no_sign_data = remove_dollar_signs(reduced_data)
    reversed_data = reverse_dates(no_sign_data)

    # Save the data to file.
    fmt = ''.join((reversed_data.shape[1] - 1) * ['%s,'] + ['%s'])
    np.savetxt(save_path, reversed_data, delimiter=',', fmt=fmt)


# ====================
# Running in Module
# ====================


if __name__ == '__main__':
    # --------------------
    # Variable Initialization
    # --------------------
    DATA_PATH = join('..', 'data', 'numerical')

    LOAD_PATH = join(DATA_PATH, 'numerical_raw.csv')
    SAVE_PATH = join(DATA_PATH, 'cleaned_data.csv')
    OLDEST_TIME = '11/26/2018'

    # --------------------
    # Main Run
    # --------------------
    main(LOAD_PATH, SAVE_PATH, OLDEST_TIME)
