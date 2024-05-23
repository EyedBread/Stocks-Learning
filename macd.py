import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def ema(data, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Start the EMA with the first value of the data

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def calculate_macd_np(prices, slow=26, fast=12, signal=9):
    # Calculate the fast and slow EMAs using the ema function
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)

    # Calculate the MACD Line
    macd_line = ema_fast - ema_slow

    # Calculate the Signal Line
    signal_line = ema(macd_line, signal)

    # Calculate the MACD Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def predict_next_day_direction(prices):
    macd_line, signal_line, histogram = calculate_macd_np(prices)
    
    # Latest points
    latest_macd = macd_line[-1]
    previous_macd = macd_line[-2]
    latest_signal = signal_line[-1]
    previous_signal = signal_line[-2]

    # MACD Crossover Logic
    if latest_macd > latest_signal and previous_macd <= previous_signal:
        return 1
    elif latest_macd < latest_signal and previous_macd >= previous_signal:
        return -1
    else:
        return 0

# Generate sample data
prices = np.random.normal(100, 0.5, size=100)

# Calculate MACD components
macd_line, signal_line, histogram = calculate_macd_np(prices)

# For example, print the first few values of each component
print("MACD Line:", macd_line[:10])
print("Signal Line:", signal_line[:10])
print("Histogram:", histogram[:10])

prediction = predict_next_day_direction(prices)
print(prediction)
