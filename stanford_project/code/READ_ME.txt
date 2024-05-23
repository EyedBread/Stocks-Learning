========================================
Purpose:
========================================
- To describe the code modules provided

========================================
Descriptions:
========================================
General -> Directory references as "data" is where the data sets were stored.
        -> Directory references as "models" is where saved models were stored.
        -> Directories not included in submission
        -> Was not sure if TAs were okay with tarred files

--------------------
Support Modules
--------------------

- text_preprocessing.py
    + Takes the raw text file and applies VADER score to each article title
    + Averages the scores over a day
    + Makes range of dates match that of numerical data
    + Assigns zeros to days with no text data
    + Change dates to match those in numerical set

- num_preprocessing.py
    + Removes $ signs from the data set
    + Limits temporal data to match the textual range
    + Inverts the time series to be oldest information first

- combining_datatypes.py
    + Adds the vader scores as a column to the numerical data
    + Removes extraneous dates, e.g., weekends from textual data

- data_prep.py
    + Takes the cleaned data and transforms it into sequences
    + If 10 days are used to predict, then shape is (m, 10, 6)
    + m is number of sequences
    + 10 is the time step size
    + 6 is the number of features
    + This creates the input to the deep learning architectures

--------------------
Deep Learning Models
--------------------

- train_single_model.py
    + Used to train and save the base case models
    + Train LSTM and GRU models on the training data

- meta_model.py
    + Used to train and predict with the validation predictions from pre-trained
      LSTM & GRU
