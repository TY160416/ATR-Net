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

from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, LayerNormalization

with tf.device('/GPU:0'):
    start_time = time.time()

    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model

            assert d_model % self.num_heads == 0

            self.depth = d_model // self.num_heads

            self.wq = Dense(d_model)
            self.wk = Dense(d_model)
            self.wv = Dense(d_model)

            self.dense = Dense(d_model)

        def split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def call(self, v, k, q, mask=None):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)
            k = self.wk(k)
            v = self.wv(v)

            q = self.split_heads(q, batch_size)
            k = self.split_heads(k, batch_size)
            v = self.split_heads(v, batch_size)

            scaled_attention, attention_weights = self.scaled_dot_product_attention(
                q, k, v, mask)

            scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
            concat_attention = tf.reshape(scaled_attention,
                                          (batch_size, -1, self.d_model))

            output = self.dense(concat_attention)
            return output, attention_weights

        def scaled_dot_product_attention(self, q, k, v, mask=None):
            matmul_qk = tf.matmul(q, k, transpose_b=True)

            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)

            return output, attention_weights


    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(TransformerBlock, self).__init__()

            self.mha = MultiHeadAttention(d_model, num_heads)
            self.ffn = tf.keras.Sequential([
                Dense(dff, activation='relu'),
                Dense(d_model)
            ])

            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)

            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)

        def call(self, x, training, mask=None):
            attn_output, attn_weights = self.mha(x, x, x, mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)

            return out2, attn_weights


    class PositionalEncoding(tf.keras.layers.Layer):
        def __init__(self, position, d_model):
            super(PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(position, d_model)

        def get_angles(self, position, i, d_model):
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return position * angles

        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model)

            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])

            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]

            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


    def Transformer_time():

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

        x = Dense(256)(inputs)

        # 位置编码
        positional_encoding = PositionalEncoding(look_back, 256)
        x = positional_encoding(x)

        # Transformer块
        transformer_block1, attn_weights1 = TransformerBlock(d_model=256, num_heads=2, dff=4, rate=0.1)(x, training=True)
        transformer_block1 = Dropout(0.01)(transformer_block1)

        x = tf.reduce_mean(transformer_block1, axis=1)


        # 输出层
        main_output = Dense(1, activation='linear', name='main_output')(x)

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

        # 输出评估结果
        print("\n" + "=" * 50)
        print("Transformer模型评估结果")
        print("=" * 50)

        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        pd.DataFrame({
            'y_true': testY_orig.flatten(),
            'y_pred': testPredict.flatten()
        }).to_csv('HSI_Transformer_predictions.csv', index=False)

    if __name__ == '__main__':
        Transformer_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")