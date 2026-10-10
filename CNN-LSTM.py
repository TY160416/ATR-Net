import numpy
import time
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf

np.random.seed(500)
numpy.set_printoptions(suppress=True)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, LSTM, MaxPooling1D

with tf.device('/GPU:0'):
    start_time = time.time()

    def CNN_LSTM_time():

        dataframe = pd.read_csv('HSI_data1.csv', usecols=['Date', 'Close'], header=0, engine='python')

        dataframe['Date'] = pd.to_datetime(dataframe['Date'])

        train_start = '2015-01-01'
        train_end = '2021-12-31'

        val_start = '2022-01-01'
        val_end = '2023-12-31'

        test_start = '2024-01-01'
        test_end = '2025-09-29'


        train_data_raw = dataframe[(dataframe['Date'] >= train_start) & (dataframe['Date'] <= train_end)]
        val_data_raw = dataframe[(dataframe['Date'] >= val_start) & (dataframe['Date'] <= val_end)]
        test_data_raw = dataframe[(dataframe['Date'] >= test_start) & (dataframe['Date'] <= test_end)]

        train_data_raw = train_data_raw['Close'].values.reshape(-1, 1)
        val_data_raw = val_data_raw['Close'].values.reshape(-1, 1)
        test_data_raw = test_data_raw['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_raw)

        train_data = scaler.transform(train_data_raw)
        val_data = scaler.transform(val_data_raw)
        test_data = scaler.transform(test_data_raw)


        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back)]
                dataX.append(a)
                dataY.append(dataset[i + look_back])
            return numpy.array(dataX), numpy.array(dataY)

        look_back = 10
        trainX, trainY = create_dataset(train_data, look_back)
        valX, valY = create_dataset(val_data, look_back)
        testX, testY = create_dataset(test_data, look_back)

        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = numpy.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        inputs = Input(shape=(look_back, 1))

        conv1 = Conv1D(filters=128, kernel_size=10, activation='relu', padding='same')(inputs)
        conv1 = Dropout(0.01)(conv1)

        lstm1 = LSTM(100, dropout=0.01)(conv1)

        main_output = Dense(1, activation='linear', name='main_output')(lstm1)

        model = Model(inputs=[inputs], outputs=[main_output])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.995,
            epsilon=1e-8
        )

        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae']
        )

        model.summary()

        history = model.fit(
            trainX, trainY,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_data=(valX, valY),
            verbose=2
        )

        trainPredict = model.predict(trainX)
        valPredict = model.predict(valX)
        testPredict = model.predict(testX)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_orig = scaler.inverse_transform(trainY.reshape(-1, 1))
        valPredict = scaler.inverse_transform(valPredict)
        valY_orig = scaler.inverse_transform(valY.reshape(-1, 1))
        testPredict = scaler.inverse_transform(testPredict)
        testY_orig = scaler.inverse_transform(testY.reshape(-1, 1))


        def rmse(predictions, targets):
            return numpy.sqrt(((predictions - targets) ** 2).mean())

        def mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))

        def mape(predictions, targets):
            return np.mean(np.abs((predictions - targets) / targets)) * 100

        def r_square(predictions, targets):
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return 1 - (ss_res / (ss_tot))

        print("\n" + "=" * 50)
        print("CNN-LSTM模型评估结果")
        print("=" * 50)

        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        pd.DataFrame({
            'y_true': testY_orig.flatten(),
            'y_pred': testPredict.flatten()
        }).to_csv('HSI_CNN_LSTM_predictions.csv', index=False)


    if __name__ == '__main__':
        CNN_LSTM_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")