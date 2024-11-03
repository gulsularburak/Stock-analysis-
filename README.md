import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Dropout, Conv1D, BatchNormalization, Input,Attention, Multiply
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import random
from lime import lime_tabular
import pandas_ta as ta
from keras.layers import Input, TimeDistributed, Flatten
from keras.layers import Attention, Multiply, Dense, LSTM, Conv1D, Dropout, Input
from keras.models import Sequential
from keras.regularizers import l2



def set_seed(seed=9):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class StockAnalysis:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock_data = None
        self.model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None

    def fetch_news_bing(self, api_key_bing):
        try:
            url = f"https://api.bing.microsoft.com/v7.0/news/search?q={self.ticker}&freshness=Day&count=100"
            headers = {"Ocp-Apim-Subscription-Key": api_key_bing}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'value' in data:
                news_contents = [article['description'] for article in data['value'] if 'description' in article]
                logger.info(f"Fetched news contents for {self.ticker} from Bing: {len(news_contents)} articles")
            else:
                logger.warning(f"No news data found for {self.ticker}")
                news_contents = []

            return news_contents
        except requests.RequestException as e:
            logger.error(f"Error fetching news data for {self.ticker} from Bing: {e}")
            return []

    def perform_sentiment_analysis(self, news_data):
        if not news_data:
            logger.warning("No news data available for sentiment analysis")
            return pd.DataFrame(columns=["description", "sentiment"])

        analyzer = SentimentIntensityAnalyzer()
        news_sentiments = []

        for description in news_data:
            sentiment_score = analyzer.polarity_scores(description)
            news_sentiments.append({
                "description": description,
                "sentiment": sentiment_score["compound"]
            })

        news_df = pd.DataFrame(news_sentiments)
        return news_df

    def download_stock_data(self, start_date, end_date):
        self.stock_data = yf.download(self.ticker, start=start_date, end=end_date)

    def fetch_bist100_data(self, start_date, end_date):
        bist100 = yf.download('XU100.IS', start=start_date, end=end_date)
        bist100 = bist100[['Close']].rename(columns={'Close': 'BIST100_Close'})
        return bist100

    def add_technical_indicators(self):
        try:
            df = self.stock_data

            start_date = df.index.min().strftime('%Y-%m-%d')
            end_date = df.index.max().strftime('%Y-%m-%d')
            bist100 = self.fetch_bist100_data(start_date, end_date)
            df = df.merge(bist100, left_index=True, right_index=True, how='left')

            # Technical Indicators using pandas_ta
            df['SMA'] = ta.sma(df['Close'], length=30)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            bollinger = ta.bbands(df['Close'], length=20)
            df['Bollinger_upper'] = bollinger['BBU_20_2.0']
            df['Bollinger_middle'] = bollinger['BBM_20_2.0']
            df['Bollinger_lower'] = bollinger['BBL_20_2.0']
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            df['Momentum'] = ta.roc(df['Close'], length=10)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ROC'] = ta.roc(df['Close'], length=10)
            df['Volume'] = df['Volume'].astype(np.int64)
            df['WilliamsR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volatility'] = df['Close'].rolling(window=14).std()
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['EMA_100'] = ta.ema(df['Close'], length=100)

            df['Volatility_Percentage'] = ((df['High'] - df['Low']) / df['Close']) * 100

            df['Z-Score'] = (df['Close'] - df['Close'].rolling(window=60).mean()) / df['Close'].rolling(window=60).std()

            correlation = df['Close'].corr(df['BIST100_Close'])
            logger.info(f"Kapanış fiyatları ile BIST100 endeksi arasındaki korelasyon: {correlation}")

            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()

            logger.info("Technical indicators, Z-Score, and lag features added, NaN values filled")
            self.stock_data = df
        except Exception as e:
            logger.error(f"Error while adding technical indicators: {e}")

    def add_behavioral_indicators(self):
        try:
            df = self.stock_data

            df['overconfidence'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['overconfidence'] = df['overconfidence'].apply(lambda x: x if x > 0 else 0)

            df['herd_behavior'] = df['RSI'].apply(lambda x: 1 if x > 70 else 0)
            df['ambiguity_aversion'] = df['ATR'].apply(lambda x: 1 if x > df['ATR'].mean() else 0)
            df['excessive_optimism'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
            df['excessive_optimism'] = df['excessive_optimism'].apply(lambda x: x if x > 0.1 else 0)
            df['loss_aversion'] = df['RSI'].apply(lambda x: 1 if x < 30 else 0)

            self.stock_data = df.ffill().bfill()
            print(self.stock_data)
            logger.info("Behavioral finance indicators added and NaN values filled")
        except Exception as e:
            logger.error(f"Error while adding behavioral indicators: {e}")

    def detect_patterns(self):
        df = self.stock_data

        # Example pattern: Bullish Engulfing
        df['Bullish_Engulfing'] = (
            (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous candle was bearish
            (df['Open'] < df['Close'].shift(1)) &  # Current candle opens below previous close
            (df['Close'] > df['Open'].shift(1))    # Current candle closes above previous open
        ).astype(int)

        # Golden Cross and Death Cross as additional examples
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['Golden_Cross'] = ((df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))).astype(int)
        df['Death_Cross'] = ((df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))).astype(int)

        df['Ascending_Triangle'] = (
                (df['Low'].rolling(window=5).min() > df['Low'].rolling(window=5).min().shift(1)) &  # Artan dipler
                (df['High'].rolling(window=5).max() == df['High'].rolling(window=5).max())  # Yatay direnç
            ).astype(int)

        df['Descending_Triangle'] = (
                (df['High'].rolling(window=5).max() < df['High'].rolling(window=5).max().shift(1)) &  # Azalan zirveler
                (df['Low'].rolling(window=5).min() == df['Low'].rolling(window=5).min())  # Yatay destek
            ).astype(int)

        df['Bullish_Flag'] = (
                (df['Close'] > df['Close'].shift(1)) &  # Güçlü yükseliş
                (df['Close'].rolling(window=5).max() < df['Close'].rolling(window=20).max()) &  # Dar bir düzeltme
                (df['Low'].rolling(window=5).min() > df['Low'].rolling(window=20).min())
            ).astype(int)



        df['Bearish_Flag'] = (
            (df['Close'] < df['Close'].shift(1)) &  # Güçlü düşüş
            (df['Close'].rolling(window=5).min() > df['Close'].rolling(window=20).min()) &  # Dar bir düzeltme
            (df['High'].rolling(window=5).max() < df['High'].rolling(window=20).max())
        ).astype(int)



        df['Rising_Wedge'] = (
            (df['High'].rolling(window=5).max() > df['High'].rolling(window=5).max().shift(1)) &  # Artan zirveler
            (df['Low'].rolling(window=5).min() > df['Low'].rolling(window=5).min().shift(1)) &  # Artan dipler
            (df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min() < df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())  # Daralan kanal
        ).astype(int)



        df['Falling_Wedge'] = (
            (df['High'].rolling(window=5).max() < df['High'].rolling(window=5).max().shift(1)) &  # Azalan zirveler
            (df['Low'].rolling(window=5).min() < df['Low'].rolling(window=5).min().shift(1)) &  # Azalan dipler
            (df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min() < df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min())  # Daralan kanal
        ).astype(int)


        df['Double_Top'] = (
            (df['High'].shift(1) > df['High'].shift(2)) &
            (df['High'].shift(1) > df['High']) &  # İki kez zirve
            (df['Close'] < df['Close'].shift(1))  # İkinci zirve sonrası düşüş
        ).astype(int)

        df['Double_Bottom'] = (
            (df['Low'].shift(1) < df['Low'].shift(2)) &
            (df['Low'].shift(1) < df['Low']) &  # İki kez dip
            (df['Close'] > df['Close'].shift(1))  # İkinci dip sonrası yükseliş
        ).astype(int)

        df['Support'] = df['Low'].rolling(window=20).min()  # Son 20 günün en düşük değeri destek olarak kabul edilir
        df['Resistance'] = df['High'].rolling(window=20).max()  # Son 20 günün en yüksek değeri direnç olarak kabul edilir

        self.stock_data = df
        self.stock_data = df.ffill().bfill()
        logger.info("Custom chart patterns detected and added to the dataset.")








    def prepare_stock_data(self, scaler_type='MinMaxScaler'):
        self.add_technical_indicators()
        self.add_behavioral_indicators()
        self.detect_patterns()

        if scaler_type == 'RobustScaler':
            self.scaler = RobustScaler()
        elif scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Tüm özellikleri ölçeklendirme
        features_to_scale = ['Open', 'High', 'Low', 'Close' ]
        scaled_data = self.scaler.fit_transform(self.stock_data[features_to_scale])
        return scaled_data, self.scaler

    def create_dataset(self, dataset):
        X, Y = [], []
        for i in range(1, len(dataset)):
            X.append(dataset[i - 1])  # Önceki günün verilerini kullanarak tahmin yapıyoruz
            Y.append(dataset[i, 3])  # 'Close' fiyatını tahmin ediyoruz (index 3)

        return np.array(X), np.array(Y)

    def prepare_lstm_data(self, scaled_stock_data):
        X, y = self.create_dataset(scaled_stock_data)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        return X_train, X_test, y_train, y_test

    def build_cnn_lstm(self, input_shape, filters=32, kernel_size=3, lstm_units_1=128, lstm_units_2=64, dropout_rate=0.3, l2_reg=0.00001, learning_rate=0.001):
        inputs = Input(shape=input_shape)

        # CNN katmanı
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='causal')(inputs)

        # LSTM katmanı (return_sequences=True ile çıktılar alınır)
        x = LSTM(units=lstm_units_1, activation='tanh', return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

        # Fully connected layer
        outputs = Dense(1, activation='linear')(x)

        # Modeli tanımla
        model = Model(inputs=inputs, outputs=outputs)

        # Modelin derlenmesi
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

        return model


    def augment_data(self, X, y, noise_factors=[0.02], scale_factors=[1.0]):
        augmented_X, augmented_y = [], []
        for i in range(len(X)):
            for noise_factor in noise_factors:
                noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(y[i])

            for scale_factor in scale_factors:
                scaled_data = X[i] * scale_factor
                augmented_X.append(scaled_data)
                augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)



    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test, scaler, epochs=100, batch_size=32):
        self.model = model
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=0)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Tahminleri geri ölçeklendirmek için sıfırlarla doldurma işlemini iyileştiriyoruz
        train_predict_full = np.zeros((train_predict.shape[0], scaler.n_features_in_))
        train_predict_full[:, 0] = train_predict[:, 0]  # Sadece 'Close' fiyatını kullanıyoruz

        test_predict_full = np.zeros((test_predict.shape[0], scaler.n_features_in_))
        test_predict_full[:, 0] = test_predict[:, 0]  # Sadece 'Close' fiyatını kullanıyoruz

        # Geri ölçeklendirme işlemi
        train_predict = scaler.inverse_transform(train_predict_full)[:, 0]
        test_predict = scaler.inverse_transform(test_predict_full)[:, 0]

        # Y değerlerinin geri ölçeklendirilmesi
        y_train_full = np.zeros((y_train.shape[0], scaler.n_features_in_))
        y_train_full[:, 0] = y_train  # Sadece 'Close' fiyatını kullanıyoruz
        y_train_inverse = scaler.inverse_transform(y_train_full)[:, 0]

        y_test_full = np.zeros((y_test.shape[0], scaler.n_features_in_))
        y_test_full[:, 0] = y_test  # Sadece 'Close' fiyatını kullanıyoruz
        y_test_inverse = scaler.inverse_transform(y_test_full)[:, 0]

        # Performans metriklerini hesaplama
        train_score_r2 = r2_score(y_train_inverse, train_predict)
        test_score_r2 = r2_score(y_test_inverse, test_predict)

        train_rmse = np.sqrt(mean_squared_error(y_train_inverse, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_inverse, test_predict))

        train_mae = mean_absolute_error(y_train_inverse, train_predict)
        test_mae = mean_absolute_error(y_test_inverse, test_predict)

        logger.info(f"Train R2 Score: {train_score_r2}")
        logger.info(f"Test R2 Score: {test_score_r2}")
        logger.info(f"Train RMSE: {train_rmse}")
        logger.info(f"Test RMSE: {test_rmse}")
        logger.info(f"Train MAE: {train_mae}")
        logger.info(f"Test MAE: {test_mae}")

        return train_score_r2, test_score_r2, train_rmse, test_rmse, train_mae, test_mae


    def predict_stock_price(self, model, X, scaler, days=5):
        predictions = []
        prediction_dates = []
        current_batch = X[-1].reshape(1, X.shape[1], X.shape[2])
        current_date = self.stock_data.index[-1]  # Tahminlerin başlangıç tarihi

        for _ in range(days):
            pred = model.predict(current_batch)[0]
            predictions.append(pred[0])  # Sadece 'Close' fiyatını tahmin ediyoruz

            # Yeni tahmin edilen fiyatı batch'e ekle ve eski değerleri kaydır
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

            # Bir sonraki tahmin tarihi
            current_date += timedelta(days=1)
            while (current_date.weekday() >= 5):  # Eğer gün hafta sonu ise
                current_date += timedelta(days=1)

            prediction_dates.append(current_date)

        # Tahminleri geri ölçeklendirme
        predictions_full = np.zeros((len(predictions), scaler.n_features_in_))
        predictions_full[:, 0] = predictions  # Sadece 'Close' fiyatını kullanıyoruz
        predicted_prices = scaler.inverse_transform(predictions_full)[:, 0]

        return list(zip(prediction_dates, predicted_prices))


    def calculate_weekly_percentage_change(self, predicted_prices):
        try:
            weekly_close = predicted_prices[::5]
            weekly_pct_change = (weekly_close[1:] - weekly_close[:-1]) / weekly_close[:-1] * 100
            logger.info(f"Weekly percentage change calculated: {weekly_pct_change}")
            return weekly_pct_change
        except Exception as e:
            logger.error(f"Error while calculating weekly percentage change: {e}")
            return None

    def plot_future_predictions(self, predictions):
        future_dates = []
        current_date = self.stock_data.index[-1]
        for _ in range(len(predictions)):
            current_date += timedelta(days=1)
            while (current_date.weekday() >= 5):  # Eğer gün hafta sonu ise
                current_date += timedelta(days=1)
            future_dates.append(current_date)

        plt.figure(figsize=(14, 7))
        plt.plot(future_dates, predictions, label='Predicted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.ticker} Future Stock Price Prediction')
        plt.legend()
        plt.show()



    def calculate_percentage_change(self, predictions):
        percentage_changes = []
        for i in range(1, len(predictions)):
            price_current = predictions[i][1]  # Mevcut günün fiyatı (tuple'ın ikinci elemanı)
            price_previous = predictions[i - 1][1]  # Önceki günün fiyatı (tuple'ın ikinci elemanı)
            percentage_change = ((price_current - price_previous) / price_previous) * 100
            percentage_changes.append(percentage_change)
        return percentage_changes


    def plot_patterns(self):
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 20), sharex=True)

        # Destek ve Direnç seviyelerini göster
        axes[0].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[0].plot(self.stock_data.index, self.stock_data['Support'], label='Support', linestyle='--', color='green')
        axes[0].plot(self.stock_data.index, self.stock_data['Resistance'], label='Resistance', linestyle='--', color='red')
        axes[0].set_title(f'{self.ticker} Support and Resistance Levels')
        axes[0].legend()

        # Bullish Engulfing
        axes[1].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[1].scatter(self.stock_data.index[self.stock_data['Bullish_Engulfing'] == 1],
                        self.stock_data['Close'][self.stock_data['Bullish_Engulfing'] == 1],
                        color='orange', label='Bullish Engulfing', marker='^', s=100)
        axes[1].set_title('Bullish Engulfing')
        axes[1].legend()

        # Golden Cross and Death Cross
        axes[2].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[2].scatter(self.stock_data.index[self.stock_data['Golden_Cross'] == 1],
                        self.stock_data['Close'][self.stock_data['Golden_Cross'] == 1],
                        color='yellow', label='Golden Cross', marker='*', s=100)
        axes[2].scatter(self.stock_data.index[self.stock_data['Death_Cross'] == 1],
                        self.stock_data['Close'][self.stock_data['Death_Cross'] == 1],
                        color='black', label='Death Cross', marker='x', s=100)
        axes[2].set_title('Golden Cross and Death Cross')
        axes[2].legend()

        # Ascending and Descending Triangle
        axes[3].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[3].scatter(self.stock_data.index[self.stock_data['Ascending_Triangle'] == 1],
                        self.stock_data['Close'][self.stock_data['Ascending_Triangle'] == 1],
                        color='purple', label='Ascending Triangle', marker='^', s=100)
        axes[3].scatter(self.stock_data.index[self.stock_data['Descending_Triangle'] == 1],
                        self.stock_data['Close'][self.stock_data['Descending_Triangle'] == 1],
                        color='cyan', label='Descending Triangle', marker='v', s=100)
        axes[3].set_title('Ascending and Descending Triangle')
        axes[3].legend()

        # Wedges (Rising and Falling)
        axes[4].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[4].scatter(self.stock_data.index[self.stock_data['Rising_Wedge'] == 1],
                        self.stock_data['Close'][self.stock_data['Rising_Wedge'] == 1],
                        color='magenta', label='Rising Wedge', marker='^', s=100)
        axes[4].scatter(self.stock_data.index[self.stock_data['Falling_Wedge'] == 1],
                        self.stock_data['Close'][self.stock_data['Falling_Wedge'] == 1],
                        color='orange', label='Falling Wedge', marker='v', s=100)
        axes[4].set_title('Rising and Falling Wedge')
        axes[4].legend()

        plt.xlabel('Date')
        plt.show()

    def plot_patterns_grouped(self):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 15), sharex=True)

        # Destek ve Direnç seviyelerini göster
        axes[0].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[0].plot(self.stock_data.index, self.stock_data['Support'], label='Support', linestyle='--', color='green')
        axes[0].plot(self.stock_data.index, self.stock_data['Resistance'], label='Resistance', linestyle='--', color='red')
        axes[0].set_title(f'{self.ticker} Support and Resistance Levels')
        axes[0].legend()

        # Formasyon Grubu 1: Engulfing, Golden Cross, Death Cross
        axes[1].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[1].scatter(self.stock_data.index[self.stock_data['Bullish_Engulfing'] == 1],
                        self.stock_data['Close'][self.stock_data['Bullish_Engulfing'] == 1],
                        color='orange', label='Bullish Engulfing', marker='^', s=100)
        axes[1].scatter(self.stock_data.index[self.stock_data['Golden_Cross'] == 1],
                        self.stock_data['Close'][self.stock_data['Golden_Cross'] == 1],
                        color='yellow', label='Golden Cross', marker='*', s=100)
        axes[1].scatter(self.stock_data.index[self.stock_data['Death_Cross'] == 1],
                        self.stock_data['Close'][self.stock_data['Death_Cross'] == 1],
                        color='black', label='Death Cross', marker='x', s=100)
        axes[1].set_title('Engulfing, Golden Cross, Death Cross')
        axes[1].legend()

        # Formasyon Grubu 2: Ascending Triangle, Descending Triangle, Wedges
        axes[2].plot(self.stock_data.index, self.stock_data['Close'], label='Close Price', color='blue')
        axes[2].scatter(self.stock_data.index[self.stock_data['Ascending_Triangle'] == 1],
                        self.stock_data['Close'][self.stock_data['Ascending_Triangle'] == 1],
                        color='purple', label='Ascending Triangle', marker='^', s=100)
        axes[2].scatter(self.stock_data.index[self.stock_data['Descending_Triangle'] == 1],
                        self.stock_data['Close'][self.stock_data['Descending_Triangle'] == 1],
                        color='cyan', label='Descending Triangle', marker='v', s=100)
        axes[2].scatter(self.stock_data.index[self.stock_data['Rising_Wedge'] == 1],
                        self.stock_data['Close'][self.stock_data['Rising_Wedge'] == 1],
                        color='magenta', label='Rising Wedge', marker='^', s=100)
        axes[2].scatter(self.stock_data.index[self.stock_data['Falling_Wedge'] == 1],
                        self.stock_data['Close'][self.stock_data['Falling_Wedge'] == 1],
                        color='orange', label='Falling Wedge', marker='v', s=100)
        axes[2].set_title('Triangles and Wedges')
        axes[2].legend()

        plt.xlabel('Date')
        plt.show()


    def plot_filtered_patterns(self, start_date=None, end_date=None):
        df = self.stock_data
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')

        # Sadece belirli bir formasyonu göstermek için (örneğin: Bullish Engulfing)
        plt.scatter(df.index[df['Bullish_Engulfing'] == 1],
                    df['Close'][df['Bullish_Engulfing'] == 1],
                    color='orange', label='Bullish Engulfing', marker='^', s=100)

        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.ticker} Filtered Patterns')
        plt.legend()
        plt.show()




    def plot_percentage_change(self, percentage_changes):
        plt.figure(figsize=(14, 7))
        plt.plot(range(1, len(percentage_changes) + 1), percentage_changes, label='Percentage Change')
        plt.xlabel('Day')
        plt.ylabel('Percentage Change (%)')
        plt.title(f'{self.ticker} Future Percentage Change')
        plt.legend()
        plt.show()

    def plot_results(self, dates, train_predict, test_predict, real_close):
        plt.figure(figsize=(14, 7))
        plt.plot(dates, real_close, label='Actual Close Price', color='blue')  # Gerçek kapanış fiyatları
        plt.plot(dates[:len(train_predict)], train_predict, label='Train Predict', color='orange')
        plt.plot(dates[len(train_predict):len(train_predict) + len(test_predict)], test_predict, label='Test Predict', color='green')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{self.ticker} Stock Price Prediction with CNN-LSTM')
        plt.legend()
        plt.show()

    def plot_sentiment(self, news_df):
        plt.figure(figsize=(14, 7))
        plt.bar(range(len(news_df)), news_df['sentiment'])
        plt.xlabel('News Article Index')
        plt.ylabel('Sentiment Score')
        plt.title('News Sentiment Scores')
        plt.show()

    def print_technical_indicators(self):
        print(self.stock_data.tail())
        print(self.stock_data.isna().sum())

    def cross_validate_model(self, scaled_stock_data, n_splits=5, epochs=100, batch_size=32):
        tscv = TimeSeriesSplit(n_splits=n_splits)

        r2_scores = []
        rmse_scores = []
        mae_scores = []

        for train_index, test_index in tscv.split(scaled_stock_data):
            train_data, test_data = scaled_stock_data[train_index], scaled_stock_data[test_index]

            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning("Not enough data for the specified split. Skipping this split.")
                continue

            try:
                X_train, y_train = self.create_dataset(train_data)
                X_test, y_test = self.create_dataset(test_data)

                if len(X_train.shape) < 2 or len(X_test.shape) < 2:
                    raise ValueError("Dataset creation failed due to insufficient data points.")

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                model = self.build_cnn_lstm_model()

                early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)

                test_predict = model.predict(X_test)
                test_predict = self.scaler.inverse_transform(test_predict)
                y_test_inverse = self.scaler.inverse_transform(y_test.reshape(-1, 1))

                r2 = r2_score(y_test_inverse, test_predict)
                rmse = np.sqrt(mean_squared_error(y_test_inverse, test_predict))
                mae = mean_absolute_error(y_test_inverse, test_predict)

                r2_scores.append(r2)
                rmse_scores.append(rmse)
                mae_scores.append(mae)

            except ValueError as ve:
                logger.warning(f"Skipping this split due to an error: {ve}")
                continue

        if len(r2_scores) == 0 or len(rmse_scores) == 0 or len(mae_scores) == 0:
            logger.warning("Not enough valid splits for cross-validation.")
            return None, None, None

        avg_r2 = np.mean(r2_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_mae = np.mean(mae_scores)

        logger.info(f"Cross-Validation R2 Score: {avg_r2}")
        logger.info(f"Cross-Validation RMSE: {avg_rmse}")
        logger.info(f"Cross-Validation MAE: {avg_mae}")

        return avg_r2, avg_rmse, avg_mae

class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def predict_function(self, data):
        data = data.reshape(-1, self.X_train.shape[1], self.X_train.shape[2])
        return self.model.predict(data).flatten()

    def explain_with_lime(self, X_test):
        if len(self.X_train.shape) != 3:
            raise ValueError("X_train must have 3 dimensions (n_samples, n_timesteps, n_features).")
        if len(X_test.shape) != 3:
            raise ValueError("X_test must have 3 dimensions (n_samples, n_timesteps, n_features).")

        n_samples, n_timesteps, n_features = self.X_train.shape

        feature_names = [f"Feature {i}" for i in range(n_timesteps * n_features)]

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.reshape(n_samples, n_timesteps * n_features),
            feature_names=feature_names,
            mode="regression",
            verbose=True
        )

        data_instance = X_test[0]

        if len(data_instance.shape) != 2 or data_instance.shape[0] != n_timesteps:
            raise ValueError(f"Data instance has shape {data_instance.shape}, expected ({n_timesteps}, {n_features}).")

        data_instance_flat = data_instance.flatten()

        print(f"Data instance shape after flattening: {data_instance_flat.shape}")

        if data_instance_flat.size != n_timesteps * n_features:
            raise ValueError(f"Data instance has {data_instance_flat.size} features, expected {n_timesteps * n_features} features.")

        exp = explainer.explain_instance(data_instance_flat, self.predict_function, num_features=10)
        exp.show_in_notebook(show_table=True, show_all=False)
        print(exp.as_list())

class StockPredictionModel:
    def __init__(self, build_model_function):
        self.build_model_function = build_model_function
        self.base_model_1 = None
        self.base_model_2 = None
        self.meta_model_instance = None

    def build_and_train_base_models(self, X_train, y_train, X_test, y_test):
        # Bu fonksiyonları self.build_model_function kullanarak oluşturuyoruz
        self.base_model_1 = self.build_model_function(filters=64, kernel_size=3, lstm_units_1=257, lstm_units_2=128, dropout_rate=0.25, l2_reg=0.00000001)
        self.base_model_2 = self.build_model_function(filters=128, kernel_size=5, lstm_units_1=257, lstm_units_2=128, dropout_rate=0.25, l2_reg=0.00000001)

        self.base_model_1.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        self.base_model_2.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        preds_1_train = self.base_model_1.predict(X_train)
        preds_2_train = self.base_model_2.predict(X_train)
        preds_1_test = self.base_model_1.predict(X_test)
        preds_2_test = self.base_model_2.predict(X_test)

        return preds_1_train, preds_2_train, preds_1_test, preds_2_test

    def build_meta_model(self):
        return LinearRegression()

    def train_and_evaluate_meta_model(self, preds_1_train, preds_2_train, preds_1_test, preds_2_test, y_train, y_test):
        train_meta_features = np.concatenate([preds_1_train, preds_2_train], axis=1)
        test_meta_features = np.concatenate([preds_1_test, preds_2_test], axis=1)

        meta_model = self.build_meta_model()
        meta_model.fit(train_meta_features, y_train)

        meta_preds_train = meta_model.predict(train_meta_features)
        meta_preds_test = meta_model.predict(test_meta_features)

        meta_train_r2 = r2_score(y_train, meta_preds_train)
        meta_test_r2 = r2_score(y_test, meta_preds_test)
        meta_train_rmse = np.sqrt(mean_squared_error(y_train, meta_preds_train))
        meta_test_rmse = np.sqrt(mean_squared_error(y_test, meta_preds_test))
        meta_train_mae = mean_absolute_error(y_train, meta_preds_train)
        meta_test_mae = mean_absolute_error(y_test, meta_preds_test)

        logger.info(f"Meta Model Eğitim R2: {meta_train_r2}")
        logger.info(f"Meta Model Test R2: {meta_test_r2}")
        logger.info(f"Meta Model Eğitim RMSE: {meta_train_rmse}")
        logger.info(f"Meta Model Test RMSE: {meta_test_rmse}")
        logger.info(f"Meta Model Eğitim MAE: {meta_train_mae}")
        logger.info(f"Meta Model Test MAE: {meta_test_mae}")

        self.meta_model_instance = meta_model  # Meta modeli kaydet

        return meta_model

    def predict_with_base_and_meta(self, X_test, y_test, scaler):
        preds_1_test = self.base_model_1.predict(X_test)
        preds_2_test = self.base_model_2.predict(X_test)

        test_meta_features = np.concatenate([preds_1_test, preds_2_test], axis=1)
        meta_preds_test = self.meta_model_instance.predict(test_meta_features)

        meta_preds_test_rescaled = scaler.inverse_transform(meta_preds_test.reshape(-1, 1))
        preds_1_test_rescaled = scaler.inverse_transform(preds_1_test)
        preds_2_test_rescaled = scaler.inverse_transform(preds_2_test)

        return meta_preds_test_rescaled, preds_1_test_rescaled, preds_2_test_rescaled

    def choose_best_predictions(self, meta_preds, preds_1, preds_2, y_test):
        r2_meta = r2_score(y_test, meta_preds)
        r2_preds_1 = r2_score(y_test, preds_1)
        r2_preds_2 = r2_score(y_test, preds_2)

        best_predictions = meta_preds  # Varsayılan olarak meta model tahminlerini seçiyoruz

        if r2_preds_1 > r2_meta and r2_preds_1 > r2_preds_2:
            best_predictions = preds_1
        elif r2_preds_2 > r2_meta and r2_preds_2 > r2_preds_1:
            best_predictions = preds_2

        return best_predictions



if __name__ == "__main__":
    ticker = "SDTTR.IS"
    start_date = "2020-01-01"
    end_date = "2024-08-12"
    api_key_bing = "baa8f5c30d9d4f69aaa911692982687c"

    set_seed(13)  # Rastgelelik için tohumlama işlemi

    analysis = StockAnalysis(ticker)

    # Verileri indir
    analysis.download_stock_data(start_date, end_date)

    # Bing üzerinden haber verilerini al
    news_data = analysis.fetch_news_bing(api_key_bing)

    # Haber verileri üzerinde duygu analizi gerçekleştir
    news_df = analysis.perform_sentiment_analysis(news_data)

    # Hisse senedi verilerini hazırlayın
    scaled_stock_data, scaler = analysis.prepare_stock_data(scaler_type='MinMaxScaler')

    # LSTM verilerini hazırlayın
    X_train, X_test, y_train, y_test = analysis.prepare_lstm_data(scaled_stock_data)
    analysis.X_train, analysis.X_test, analysis.y_train, analysis.y_test = X_train, X_test, y_train, y_test

    # Eğitim verilerini artırın (Augmenting)
    X_train_augmented, y_train_augmented = analysis.augment_data(X_train, y_train)

    # CNN-LSTM modelini oluştur (Attention katmanı olmadan)
    input_shape = (X_train.shape[1], X_train.shape[2])  # Örneğin (time_steps, features)
    model = analysis.build_cnn_lstm(input_shape)

    # Modeli eğit ve değerlendir
    train_score_r2, test_score_r2, train_rmse, test_rmse, train_mae, test_mae = analysis.train_and_evaluate_model(
    model, X_train_augmented, y_train_augmented, X_test, y_test, scaler, epochs=10, batch_size=64)

    print(f"Train R2 Score: {train_score_r2}")
    print(f"Test R2 Score: {test_score_r2}")
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")
    print(f"Train MAE: {train_mae}")
    print(f"Test MAE: {test_mae}")

    # Gelecek 5 gün için hisse senedi fiyat tahminini yap
    future_predict = analysis.predict_stock_price(model, X_test, scaler, days=5)
    print(f"Next 5 days stock price prediction: {future_predict}")

    # Gelecek tahminleri çizin
    analysis.plot_future_predictions([price for date, price in future_predict])

    # Yüzde değişimini hesaplayın ve çizin
    percentage_changes = analysis.calculate_percentage_change(future_predict)
    analysis.plot_percentage_change(percentage_changes)

    # Teknik göstergeleri ve olası eksik değerleri yazdırın
    analysis.print_technical_indicators()


    input_shape = (X_train.shape[1], X_train.shape[2])
    meta_model = StockPredictionModel(lambda **kwargs: analysis.build_cnn_lstm_with_attention(input_shape, **kwargs))
    preds_1_train, preds_2_train, preds_1_test, preds_2_test = meta_model.build_and_train_base_models(X_train, y_train, X_test, y_test)
    meta_model_instance = meta_model.train_and_evaluate_meta_model(preds_1_train, preds_2_train, preds_1_test, preds_2_test, y_train, y_test)




    # Meta model predictions
    meta_preds_test_rescaled, preds_1_test_rescaled, preds_2_test_rescaled = meta_model.predict_with_base_and_meta(X_test, y_test, close_scaler)
    best_predictions = meta_model.choose_best_predictions(meta_preds_test_rescaled, preds_1_test_rescaled, preds_2_test_rescaled, close_scaler.inverse_transform(y_test.reshape(-1, 1)))

    # Future predictions with meta model
    future_predict_meta = meta_model_instance.predict(np.concatenate([preds_1_test[-1:], preds_2_test[-1:]], axis=1))
    future_predict_meta_rescaled = close_scaler.inverse_transform(future_predict_meta.reshape(-1, 1))
    print(f"Next 5 days stock price prediction (Meta Model): {future_predict_meta_rescaled}")

    # Verilerin son birkaç satırını kontrol et
    print("Verilerin son 5 satırı:")
    print(analysis.stock_data.tail())

    # Son kapanış fiyatını kontrol et
    last_close_price = analysis.stock_data['Close'].iloc[-1]
    print(f"Son kapanış fiyatı: {last_close_price}")



    # Modelin tahmin yaptığı kodu kontrol et
    test_predictions = model.predict(analysis.X_test)
    test_predictions_rescaled = close_scaler.inverse_transform(test_predictions)
    print(f"Tahminlerin ilk 5 değeri: {test_predictions_rescaled[:5]}")

    # Son kapanış fiyatı ve ilk tahmin fiyatını karşılaştır
    print(f"Son kapanış fiyatı: {last_close_price}")
    print(f"İlk tahmin fiyatı: {test_predictions_rescaled[0][0]}")

    # Verilerin son durumu ve eksikliklerin kontrolü
    print("Verilerin son 5 satırı:")
    print(analysis.stock_data.tail())

    print("Eksik değerlerin sayısı:")
    print(analysis.stock_data.isna().sum())

    # Hata dağılım grafiği
    plt.figure(figsize=(14, 7))
    errors = test_predictions_rescaled[:len(y_test)] - close_scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.show()

    # LIME açıklama ekleme
    model_explainer = ModelExplainer(model, X_train)
    model_explainer.explain_with_lime(X_test)
