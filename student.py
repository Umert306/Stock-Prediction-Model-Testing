# Library Imports
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

np.random.seed(40496259)

# PLS Class
class PLSTransformer(PLSRegression):
    """
    Wrapper for PLSRegression to be used in a pipeline.
    Ensures fit_transform and transform only return X_scores (2D array),
    even when y is passed during pipeline.fit().
    """
    def fit(self, X, y=None):
        return super().fit(X, y)

    def transform(self, X, y=None):
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Student Class
class Student:
    """
    Features: Log returns, Momentum, Volatility, RSI, SMA Dist, Lags, MACD, BBands, Volume Ratio
    Feature Extraction: PLS
    Model: Lasso regression.
    """

    def __init__(
        self,
        config=None,
        random_state: int = 40496259,
        *,
        # Feature arameters
        mom_windows=(5, 10, 20),
        vol_windows=(5, 10, 20),
        rsi_window=14,
        sma_windows=(10, 20),
        lags=(1, 2, 3, 5),
        macd_short=12,
        macd_long=26,
        macd_signal=9,
        bb_window=20,
        bb_num_std=2,
        vr_window=20,
        pls_components=5,
        lasso_alpha=0.01,
        min_train_points=200 
    ):
        self.random_state = int(random_state)
        np.random.seed(self.random_state)

        # Store all hyperparameters
        self.mom_windows = mom_windows
        self.vol_windows = vol_windows
        self.rsi_window = rsi_window
        self.sma_windows = sma_windows
        self.lags = lags
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal = macd_signal
        self.bb_window = bb_window
        self.bb_num_std = bb_num_std
        self.vr_window = vr_window
        
        self.pls_components = pls_components
        self.lasso_alpha = lasso_alpha
        
        self.min_train_points = min_train_points

        # Apply config overrides
        if isinstance(config, dict):
            for k, v in config.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        # This will hold the entire sklearn pipeline
        self.pipe_ = None 
        
        self.fitted_ = False
        self.fallback_pipe_ = None 

    # Static Helpers Methodss
    @staticmethod
    def _close_series(X: pd.DataFrame) -> pd.Series:
        return X["Close"] if "Close" in X.columns else X.iloc[:, 0]

    @staticmethod
    def _rsi(close: pd.Series, window: int) -> pd.Series:
        close = pd.Series(close).astype(float)
        diff = close.diff()
        gain = diff.clip(lower=0.0)
        loss = -diff.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 1 - (1 / (1 + rs))  
        return rsi.fillna(0.5)

    @staticmethod
    def _finite_mean(y: pd.Series) -> float:
        yv = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(yv) == 0:
            return 0.0
        m = float(yv.mean())
        return m if np.isfinite(m) else 0.0

    def _compute_macd(self, close_series: pd.Series):
        ema_short = close_series.ewm(span=self.macd_short, adjust=False).mean()
        ema_long = close_series.ewm(span=self.macd_long, adjust=False).mean()
        macd = ema_short - ema_long
        signal_line = macd.ewm(span=self.macd_signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist


    # Feature Engineering 

    def _make_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Leakage-safe features from a single-ticker OHLCV DataFrame.
        """
        X = X.sort_index()
        
        feats = {}
        close = self._close_series(X).astype(float)
        volume = X["Volume"].astype(float) if "Volume" in X.columns else pd.Series(0, index=X.index)

        # Log Return
        log_return = np.log(close / close.shift(1))
        feats['log_return'] = log_return

        # Momentum and Volatility
        for w in self.mom_windows:
            feats[f'mom_{w}'] = log_return.rolling(window=w, min_periods=w).mean()
        for w in self.vol_windows:
            feats[f'vol_{w}'] = log_return.rolling(window=w, min_periods=w).std()

        # RSI
        feats['rsi_14'] = self._rsi(close, self.rsi_window)

        # SMA Distance
        for w in self.sma_windows:
            sma = close.rolling(w, min_periods=w).mean()
            feats[f'sma_dist_{w}'] = (close - sma) / sma.replace(0, np.nan)

        # Lags
        for lag in self.lags:
            feats[f'lag_{lag}'] = log_return.shift(lag)

        # MACD
        feats['macd'], feats['macd_signal'], feats['macd_hist'] = self._compute_macd(close)

        # Bollinger Bands (using mom_20 and vol_20 if available)
        if self.bb_window in self.mom_windows and self.bb_window in self.vol_windows:
            if f'mom_{self.bb_window}' in feats and f'vol_{self.bb_window}' in feats:
                rolling_mean = feats[f'mom_{self.bb_window}']
                rolling_std = feats[f'vol_{self.bb_window}']
                feats['bb_upper'] = rolling_mean + (self.bb_num_std * rolling_std)
                feats['bb_lower'] = rolling_mean - (self.bb_num_std * rolling_std)
                feats['bb_bandwidth'] = (feats['bb_upper'] - feats['bb_lower']) / rolling_mean.replace(0, np.nan)

        # Volume Rato
        diff = close.diff()
        up_vol = np.where(diff > 0, volume, 0)
        down_vol = np.where(diff < 0, volume, 0)
        
        up_vol_rolling = pd.Series(up_vol, index=X.index).rolling(self.vr_window).sum()
        down_vol_rolling = pd.Series(down_vol, index=X.index).rolling(self.vr_window).sum()
        
        feats['Volume_Ratio'] = up_vol_rolling / down_vol_rolling.replace(0, np.nan)

        # Combine, replace infs, and DROPNA
        F = pd.DataFrame(feats, index=X.index)
        F = F.replace([np.inf, -np.inf], np.nan)
        F = F.dropna() 
        return F



    # Fit Method
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, meta=None):
        """
        Fit the PLS + Lasso pipeline.
        """
        np.random.seed(self.random_state)

        # 1 Create features
        F = self._make_features(X_train)
        
        # 2 Store fallback mean 
        mean_y = self._finite_mean(y_train)
        self.fallback_pipe_ = Pipeline([
            ("model", DummyRegressor(strategy="constant", constant=mean_y))
        ])
        self.fallback_pipe_.fit([[0.0]], [mean_y])

        # 3 Align features and target (REGRESSION target)
        y = y_train.reindex(F.index)
        mask = y.replace([np.inf, -np.inf], np.nan).notna()
        F_clean, y_clean = F.loc[mask], y.loc[mask]

        # 4 Handle short history (fallback to mean)
        if len(F_clean) < self.min_train_points:
            self.fitted_ = True 
            return self

        # 5 Define and Fit the new pipeline
        self.pipe_ = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSTransformer(n_components=self.pls_components)),
            ("model", Lasso(alpha=self.lasso_alpha, random_state=self.random_state))
        ])
        
        self.pipe_.fit(F_clean.values, y_clean.values)

        self.fitted_ = True
        return self
    


    # Predict Method
    def predict(self, X: pd.DataFrame, meta=None) -> pd.Series:
        """
        Return predictions from the fitted PLS + Lasso pipeline.
        """
        # 1 Check if fitted
        if not self.fitted_ or self.fallback_pipe_ is None or self.pipe_ is None:
            # Not fitted or in a fallback state from a short fit()
            if self.fallback_pipe_ is None:
                 return pd.Series(0.0, index=X.index, name="y_pred")
            # Use the fallback dummy regressor
            y_hat = self.fallback_pipe_.predict(X.iloc[:, :1]) 
            return pd.Series(y_hat, index=X.index, name="y_pred")

        # 2 Create features for the prediction set
        F = self._make_features(X)

        if F.empty:
            y_hat = self.fallback_pipe_.predict(X.iloc[:, :1])
            return pd.Series(y_hat, index=X.index, name="y_pred")

        # 3 Apply trained pipeline
        y_hat_values = self.pipe_.predict(F.values)

        # 4 Create final Series (only for valid feature dates)
        y_pred = pd.Series(y_hat_values, index=F.index, name="y_pred")
        
        # 5 Reindex to full X index, filling non-predicted values with fallback
        y_pred_full = y_pred.reindex(X.index, fill_value=self.fallback_pipe_.steps[-1][1].constant)
        y_pred_full.name = "y_pred"
        
        return y_pred_full
