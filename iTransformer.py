import numpy
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf

np.random.seed(500)
numpy.set_printoptions(suppress=True)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, LayerNormalization, MultiHeadAttention

with tf.device('/GPU:0'):
    start_time = time.time()

    class RevinNorm(tf.keras.layers.Layer):
        """Reversible Instance Normalization"""

        def __init__(self, affine=True):
            super(RevinNorm, self).__init__()
            self.affine = affine
            if self.affine:
                self.affine_weight = None
                self.affine_bias = None

        def build(self, input_shape):
            if self.affine:
                self.affine_weight = self.add_weight(
                    name='affine_weight',
                    shape=(input_shape[-1],),
                    initializer='ones'
                )
                self.affine_bias = self.add_weight(
                    name='affine_bias',
                    shape=(input_shape[-1],),
                    initializer='zeros'
                )
            super(RevinNorm, self).build(input_shape)

        def call(self, x, mode='norm'):
            if mode == 'norm':

                self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
                self.stdev = tf.sqrt(tf.reduce_mean(tf.square(x - self.mean), axis=1, keepdims=True) + 1e-5)
                x = (x - self.mean) / self.stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
                return x
            elif mode == 'denorm':

                if self.affine:
                    x = (x - self.affine_bias) / self.affine_weight
                x = x * self.stdev + self.mean
                return x


    class iTransformerEncoderLayer(tf.keras.layers.Layer):

        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
            super(iTransformerEncoderLayer, self).__init__()
            d_ff = d_ff or 4 * d_model
            self.d_model = d_model
            self.n_heads = n_heads


            self.self_attention = MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model // n_heads,
                dropout=dropout
            )
            self.dropout1 = Dropout(dropout)
            self.norm1 = LayerNormalization(epsilon=1e-6)


            self.ffn = tf.keras.Sequential([
                Dense(d_ff, activation='relu'),
                Dropout(dropout),
                Dense(d_model),
                Dropout(dropout)
            ])
            self.norm2 = LayerNormalization(epsilon=1e-6)

        def call(self, x, training=False):

            attn_output = self.self_attention(x, x, training=training)
            x = x + self.dropout1(attn_output, training=training)
            x = self.norm1(x)


            ffn_output = self.ffn(x, training=training)
            x = x + ffn_output
            x = self.norm2(x)

            return x


    class iTransformerModel(tf.keras.Model):

        def __init__(self, seq_len=10, pred_len=1, d_model=64, n_heads=2, e_layers=2, d_ff=128,
                     dropout=0.1, revin=True):
            super(iTransformerModel, self).__init__()
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.d_model = d_model
            self.revin = revin


            if revin:
                self.revin_norm = RevinNorm(affine=True)

            self.var_projection = Dense(d_model)

            self.encoder_layers = []
            for _ in range(e_layers):
                encoder_layer = iTransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout
                )
                self.encoder_layers.append(encoder_layer)


            self.prediction_head = tf.keras.Sequential([
                Dense(d_ff, activation='relu'),
                Dropout(dropout),
                Dense(pred_len)
            ])

        def call(self, x, training=False):

            if self.revin:
                x = self.revin_norm(x, mode='norm')

            x_proj = self.var_projection(x)  # [B, L, d_model]

            encoder_output = x_proj
            for encoder_layer in self.encoder_layers:
                encoder_output = encoder_layer(encoder_output, training=training)


            pooled = tf.reduce_mean(encoder_output, axis=1)  # [B, d_model]


            pred = self.prediction_head(pooled)  # [B, pred_len]
            pred = tf.expand_dims(pred, -1)  # [B, pred_len, 1]

            if self.revin:
                pred = self.revin_norm(pred, mode='denorm')

            return pred


    def iTransformer_time():

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

        train_data_raw_values = train_data_raw['Close'].values.reshape(-1, 1)
        val_data_raw_values = val_data_raw['Close'].values.reshape(-1, 1)
        test_data_raw_values = test_data_raw['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_raw_values)

        train_data = scaler.transform(train_data_raw_values)
        val_data = scaler.transform(val_data_raw_values)
        test_data = scaler.transform(test_data_raw_values)


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

        model = iTransformerModel(
            seq_len=look_back,
            pred_len=1,
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_ff=64,
            dropout=0.01,
            revin=True
        )

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

        _ = model(tf.convert_to_tensor(trainX[:32], dtype=tf.float32))

        model.summary()

        history = model.fit(
            trainX, trainY,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_data=(valX, valY),
            verbose=2
        )

        # 预测
        trainPredict = model.predict(trainX)
        valPredict = model.predict(valX)
        testPredict = model.predict(testX)

        trainPredict = trainPredict.reshape(-1, 1)
        valPredict = valPredict.reshape(-1, 1)
        testPredict = testPredict.reshape(-1, 1)
        trainY = trainY.reshape(-1, 1)
        valY = valY.reshape(-1, 1)
        testY = testY.reshape(-1, 1)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_orig = scaler.inverse_transform(trainY)
        valPredict = scaler.inverse_transform(valPredict)
        valY_orig = scaler.inverse_transform(valY)
        testPredict = scaler.inverse_transform(testPredict)
        testY_orig = scaler.inverse_transform(testY)


        def rmse(predictions, targets):
            return numpy.sqrt(((predictions - targets) ** 2).mean())

        def mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))

        def mape(predictions, targets):
            return np.mean(np.abs((predictions - targets) / targets)) * 100

        def r_square(predictions, targets):
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))

        # 输出评估结果
        print("\n" + "=" * 50)
        print("iTransformer模型评估结果")
        print("=" * 50)


        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        pd.DataFrame({
            'y_true': testY_orig.flatten(),
            'y_pred': testPredict.flatten()
        }).to_csv('HSI_iTransformer_predictions.csv', index=False)


    if __name__ == '__main__':
        iTransformer_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")