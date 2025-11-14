# Financial Time-Series Forecasting System

This project implements a **leakage-safe financial time-series forecasting pipeline** using PLS feature extraction and Lasso regression. It predicts **next-H-day cumulative log returns** for multiple stock tickers using engineered technical features. The system is designed to be robust, interpretable, and adaptive to different tickers and timeframes.

---

## Description

The pipeline constructs features such as **momentum, volatility, RSI, SMA distances, MACD components, Bollinger Bands, and Volume Ratio**, ensuring **no future data leakage**. Dimensionality is reduced using **PLS (Partial Least Squares)**, and predictions are made using **Lasso regression**. For small datasets, a **fallback dummy regressor** predicts the mean return.  

The project includes a **Jupyter Notebook** that evaluates multiple models in a **leakage-free walk-forward method**, allowing realistic assessment of **directional accuracy, MAE, and RMSE** across tickers. The evaluation highlights the benefits of sequential modeling and proper feature handling.

The **dataset** consists of historical OHLCV stock prices, and the `student.py` module provides a reusable class `Student` to compute engineered features, apply PLS, and fit Lasso models with a consistent API.

---

## Features

* **PLS Feature Extraction:** Reduces dimensionality while retaining maximum covariance with the target.
* **Lasso Regression:** Selects relevant features and prevents overfitting.
* **Leakage-Safe Feature Engineering:** Only uses past data to compute technical indicators.
* **Fallback Strategy:** Handles short training histories with a mean-predicting dummy model.
* **Technical Indicators:** Momentum, Volatility, RSI, SMA distances, MACD, Bollinger Bands, Volume Ratio.
* **Walk-Forward Evaluation:** Realistic out-of-sample testing with sequential train-test splits.

---

## Installation

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
