'''
Module Name: meta_model.py
Author: Dylan M. Crain

Description: Performs the Blended Ensemble model training and prediction.
'''


# ====================
# Libray Imports
# ====================


from os.path import join
import numpy as np

from keras.models import load_model, Model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from data_prep import data_prep


# ====================
# Supporting Defs
# ====================


def collect_data(data_path, data_params, test=True):
    '''
    Function
    --------
    collect_data():
        Gathers the sequences of data used for validation and testing.

    Parameters
    ----------
    data_path: str
        Path to the data file.
    data_params: tuple
        Composed of (horizon, days_forward, end_split).
    test: bool
        If true, the test data is returned, otherwise, the validation.

    Returns
    -------
    : tuple
        Either the test or validation data.
    '''
    horizon, days_forward, end_split = data_params
    split_y, split_X = data_prep(data_path, horizon, days_forward, end_split)

    _, validate_y, test_y = split_y
    _, validate_X, test_X = split_X

    if test:
        return (test_y, test_X)
    else:
        return (validate_y, validate_X)


def retrieve_meta_data(data_path, data_params, model_paths, test):
    '''
    Function
    --------
    retriev_meta_data():
        Runs the data set through the pre-trained LSTM and GRU models and
        concatenates the predictions to form a new data set.

    Parameters
    ----------
    data_path: str
        Path to the data file.
    data_params: tuple
        Composed of (horizon, days_forward, end_split).
    model_paths: dict
        Dictionary of paths to LSTM and GRU models.
    test: bool
        If true, the test data is returned, otherwise, the validation.

    Returns
    -------
    y: ndarray
        True stock movement for the data set: 1 for upwards and 0 for
        downwards.
    new_X: ndarray
        2D array of shape (sequences X time steps X features).
    '''
    # Collect initial data
    y, X = collect_data(data_path, data_params, test)

    # Load base models
    lstm_model = load_model(model_paths['lstm'])
    gru_model = load_model(model_paths['gru'])

    # Predict from the base models
    lstm_pred = lstm_model.predict(X).reshape(-1, 1)
    gru_pred = gru_model.predict(X).reshape(-1, 1)

    # Form and return new data set
    new_X = np.hstack((lstm_pred, gru_pred))

    return y, new_X


def meta_model(num_models):
    '''
    Function
    --------
    meta_model():
        Builds the meta model of dense layers.

    Parameters
    ----------
    num_models: int
        Number of base models used for the result: 2 (LSTM & GRU).

    Returns
    -------
    model: keras.models.Model
        Meta portion of the model that takes as input predictions from the base
        models.
    '''
    inputs = Input(shape=(num_models,))

    X = Dense(units=30, activation='relu')(inputs)
    X = Dense(units=25, activation='relu')(X)
    X = Dense(units=20, activation='relu')(X)

    X = Dense(units=1, activation='sigmoid')(X)

    model = Model(inputs=inputs, outputs=X)
    return model


# ====================
# Main Defs
# ====================


def train_meta_model(data_path, data_params, model_paths, save_path):
    '''
    Function
    --------
    train_meta_model():
        Train the BE model on the validation predictions from the base models.

    Parameters
    ----------
    data_path: str
        Path to the data file.
    data_params: tuple
        Composed of (horizon, days_forward, end_split).
    model_paths: dict
        Dictionary of paths to LSTM and GRU models.
    save_path: str
        Path to save the BE model to after training.

    Returns
    -------
    : None
        Saves the trained model to save_path.
    '''
    validate_y, validate_X = retrieve_meta_data(data_path, data_params,
                                                model_paths, test=False)
    model = meta_model(validate_X.shape[1])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.fit(validate_X, validate_y, batch_size=8, epochs=100)
    model.save(save_path + ".keras")


def predict_meta_model(meta_load, data_path, data_params, model_paths):
    '''
    Function
    --------
    predict_meta_model():
        Uses the trained BE model to predict the test set.

    Parameters
    ----------
    meta_load: str
        Path to the BE model.
    data_path: str
        Path to the data file.
    data_params: tuple
        Composed of (horizon, days_forward, end_split).
    model_paths: dict
        Dictionary of paths to LSTM and GRU models.
    '''
    test_y, test_X = retrieve_meta_data(data_path, data_params,
                                        model_paths, test=True)
    model = load_model(meta_load)
    model.evaluate(test_X, test_y)
    # print the accuracy, precision, recall, and F1 score
    print(test_X)
    print()
    output = model.predict(test_X)
    print(output)





# ====================
# Running from Module
# ====================


if __name__ == '__main__':
    # --------------------
    # Variable initialization
    # --------------------
    DATA_PATH = join('..', 'data', 'combined', 'amzn_source_price_2017-2020.csv')
    DATA_PARAMS = (10, 1, 40)  # (horizon, days_forward, end_split)
    MODEL_PATHS = {'lstm': join('..', 'models', 'lstm_one_day.keras'),
                   'gru': join('..', 'models', 'gru_one_day.keras')}
    SAVE_PATH = join('..', 'models', 'meta_one_day.keras')

    # --------------------
    # Train BE & Predict
    # --------------------
    train_meta_model(DATA_PATH, DATA_PARAMS, MODEL_PATHS, SAVE_PATH)
    predict_meta_model(SAVE_PATH, DATA_PATH, DATA_PARAMS, MODEL_PATHS)
