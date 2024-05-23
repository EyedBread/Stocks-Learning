'''
Module Name: combining_datatypes.py
Author: Dylan M. Crain

Description: This module combines the numerical (stock data) along with the
             textual (VADER scores from news articles) information into one
             data structure.
'''


# ====================
# Load Libraries
# ====================


from os.path import join
import numpy as np


# ====================
# Supporting Defs
# ====================


def gather_file(load_path):
    '''
    Function
    --------
    gather_file():
        Gather data from a given cleaned up data file.

    Parameters
    ----------
    load_path: str
        Path to file to load in.

    Returns
    -------
    data: ndarray
        Data from importing the file of interest.
    '''
    data = np.genfromtxt(load_path, delimiter=',', dtype=str)
    return data


def extract_vader_scores(text_data):
    '''
    Function
    --------
    extract_vader_scores():
        Specifically extracts the scores from the articles of interest.

    Parameters
    ----------
    text_data: ndarray
        2D array of columns date and VADER score.

    Returns
    -------
    : ndarray
        ndarray of only the VADER scores.
    '''
    return text_data[:, 1]


def combine_data(num_data, scores, text_data):
    '''
    Function
    --------
    combine_data():
        Combines the data from the numeric and textural data.

    Parameters
    ----------
    num_data: ndarray
        Data from the numerical stores.
    scores: ndarray
        Only the scores for the textual data.
    text_data: ndarray
        Data from the textual source.

    Returns
    -------
    combined_data: ndarray
        ndarray with one more column than num_data that includes the scores of
        the textual data.
    '''
    # Initialize full data structure.
    rows = num_data.shape[0]
    cols = num_data.shape[1] + 1
    combined_data = np.zeros((rows, cols)).astype(str)

    # Place old num_data into the first n-1 columns.
    combined_data[:, :-1] = num_data
    combined_data[0, -1] = 'Scores'

    # For matching dates, add the VADER scores.
    for index, date in enumerate(num_data[1:, 0]):
        loc = np.argwhere(text_data[:, 0] == date)[0][0]
        score = scores[loc]

        combined_data[index + 1, -1] = score

    return combined_data


def save_data(save_path, combined_data):
    '''
    Function
    --------
    save_data():
        Saves the final data structure to file.

    Parameters
    ----------
    save_path: str
        Path to save the data structure to.
    combined_data: ndarray
        Data structure that contains numeric and textual data.

    Returns
    -------
    : None
        Saves the data structure to the proposed file.
    '''
    fmt = ''.join((combined_data.shape[1] - 1) * ['%s,'] + ['%s'])
    np.savetxt(save_path, combined_data, delimiter=',', fmt=fmt)


# ====================
# Main Def
# ====================


def main(text_path, num_path, save_path):
    '''
    Function
    --------
    main():
        Combines the data arrays into one setting.

    Parameters
    ----------
    text_path: str
        Path to the textual data.
    num_path: str
        Path to the numerical data.
    save_path: str
        Path to save the combined data to.

    Returns
    -------
    : None
        Combines the data structures and saves it.
    '''
    # Gather both data structures.
    text_data = gather_file(text_path)
    num_data = gather_file(num_path)

    # Combine them into one.
    scores = extract_vader_scores(text_data)
    combined_data = combine_data(num_data, scores, text_data)

    # Save the result.
    save_data(save_path, combined_data)


# ====================
# Run from Module
# ====================


if __name__ == '__main__':
    # --------------------
    # Initialize Variables
    # --------------------
    TEXT_PATH = join('..', 'data', 'textual', 'cleaned_data.csv')
    NUM_PATH = join('..', 'data', 'numerical', 'cleaned_data.csv')
    SAVE_PATH = join('..', 'data', 'combined', 'cleaned_data.csv')

    # --------------------
    # Run main
    # --------------------
    main(TEXT_PATH, NUM_PATH, SAVE_PATH)
