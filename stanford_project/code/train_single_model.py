'''
Module Name: train_single_model.py
Author: Dylan M. Crain

Description: Trains the stock data on movement of the stock at a future time
             using either an LSTM or GRU setup.
'''


# ====================
# Import Libraries
# ====================


from os.path import join
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, GRU
from tensorflow.keras.optimizers import RMSprop

from data_prep import data_prep


# ====================
# Model Defs
# ====================


def build_LSTM_model(horizon, feature_size):
    '''
    Function
    --------
    build_LSTM_model():
        Builds a layered LSTM model to predict stock movement.

    Parameters
    ----------
    horizon: int
        Number of previous days used to predict per time-step.
    feature_size: int
        Number of independent parameters at each time-step.

    Returns
    -------
    model: keras.models.Model
        Model of layered LSTM that will predict stock movement.
    '''
    input_sequence = Input(shape=(horizon, feature_size))

    X = LSTM(units=60, return_sequences=True)(input_sequence)
    X = Dropout(rate=0.4)(X)
    X = LSTM(units=55, return_sequences=True)(X)
    X = Dropout(rate=0.4)(X)
    X = LSTM(units=50, return_sequences=True)(X)
    X = Dropout(rate=0.4)(X)
    X = LSTM(units=45, return_sequences=False)(X)
    X = Dropout(rate=0.4)(X)

    X = Dense(units=1, activation='sigmoid')(X)

    model = Model(inputs=input_sequence, outputs=X)
    return model


def build_GRU_model(horizon, feature_size):
    '''
    Function
    --------
    build_GRU_model():
        Builds a layered GRU model to predict stock movement.

    Parameters
    ----------
    horizon: int
        Number of previous days used to predict per time-step.
    feature_size: int
        Number of independent parameters at each time-step.

    Returns
    -------
    model: keras.models.Model
        Model of layered GRU that will predict stock movement.
    '''
    input_sequence = Input(shape=(horizon, feature_size))

    X = GRU(units=60, return_sequences=True)(input_sequence)
    X = Dropout(rate=0.2)(X)
    X = GRU(units=55, return_sequences=True)(X)
    X = Dropout(rate=0.2)(X)
    X = GRU(units=50, return_sequences=True)(X)
    X = Dropout(rate=0.2)(X)
    X = GRU(units=45, return_sequences=False)(X)
    X = Dropout(rate=0.2)(X)

    X = Dense(units=1, activation='sigmoid')(X)

    model = Model(inputs=input_sequence, outputs=X)
    return model


# ====================
# Train Model Def
# ====================


def train_single_model(data_path, data_params, epochs, lstm=True, save_path=None):
    '''
    Function
    --------
    train_single_model():
        Fits a single (LSTM or GRU) model on the stock data to predict stock
        price movements.

    Parameters
    ----------
    data_path: str
        Path to the cleaned data.
    data_params: tuple
        Tuple of "horizon", "days_forward", and "end_split".
    lstm: bool
        Flag that, when true, fits and LSTM model, GRU model otherwise.
    save_path: str
        Path to save the model to.

    Returns
    -------
    : None
        Saves the fit model to file.
    '''
    # Unravel the data_params and extract data-set.
    horizon, days_forward, end_split = data_params
    split_y, split_X = data_prep(data_path, horizon, days_forward, end_split)

    # Unravel training data and get feature size.
    y_train, y_validate, y_test = split_y
    X_train, X_validate, X_test = split_X
    feature_size = X_train.shape[-1]

    # Build the DL network and return the model.
    if lstm:
        model = build_LSTM_model(horizon, feature_size)
    else:
        model = build_GRU_model(horizon, feature_size)

    # Compile and fit
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=0.0008),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=16, epochs=epochs, shuffle=True,
              validation_data=(X_validate, y_validate))
    model.save(save_path + ".keras")


# ====================
# Running from Main
# ====================


if __name__ == '__main__':
    # --------------------
    # Variable Initializer
    # --------------------
    DATA_PATH = join('..', 'data', 'combined', 'amzn_source_price_2017-2020.csv')
    GRU_SAVE = join('..', 'models', 'gru_one_day')
    LSTM_SAVE = join('..', 'models', 'lstm_one_day')

    HORIZON = 10
    DAYS_FORWARD = 1
    END_SPLIT = 40
    EPOCHS = 100 

    DATA_PARAMS = (HORIZON, DAYS_FORWARD, END_SPLIT)

    # --------------------
    # Fit Models
    # --------------------
    train_single_model(DATA_PATH, DATA_PARAMS, EPOCHS, lstm=True,
                       save_path=LSTM_SAVE)
