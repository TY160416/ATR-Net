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

from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, LayerNormalization

with tf.device('/GPU:0'):
    start_time = time.time()

    class Transpose(tf.keras.layers.Layer):
        def __init__(self, dims, **kwargs):
            super(Transpose, self).__init__(**kwargs)
            self.dims = dims

        def call(self, x):
            return tf.transpose(x, perm=self.dims)

        def get_config(self):
            config = super(Transpose, self).get_config()
            config.update({'dims': self.dims})
            return config

    class PatchEmbedding(tf.keras.layers.Layer):
        def __init__(self, d_model, patch_len, stride, padding, dropout_rate=0.1, **kwargs):
            super(PatchEmbedding, self).__init__(**kwargs)
            self.patch_len = patch_len
            self.stride = stride
            self.d_model = d_model
            self.dropout_rate = dropout_rate

            self.conv = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=patch_len,
                strides=stride,
                padding='same' if padding else 'valid',
                use_bias=False
            )
            self.dropout = tf.keras.layers.Dropout(dropout_rate)

        def call(self, x, training=False):
            batch_size = tf.shape(x)[0]
            n_vars = tf.shape(x)[1]

            x_reshaped = tf.reshape(x, [batch_size * n_vars, -1, 1])

            x_patched = self.conv(x_reshaped)

            x_patched = self.dropout(x_patched, training=training)

            return x_patched, n_vars


    class MultiHeadAttention(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dropout_rate=0.1):
            super(MultiHeadAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model
            self.depth = d_model // num_heads

            self.wq = Dense(d_model)
            self.wk = Dense(d_model)
            self.wv = Dense(d_model)

            self.dense = Dense(d_model)
            self.dropout = Dropout(dropout_rate)

        def call(self, v, k, q, mask=None, training=False):
            batch_size = tf.shape(q)[0]

            q = self.wq(q)  # (batch_size, seq_len, d_model)
            k = self.wk(k)  # (batch_size, seq_len, d_model)
            v = self.wv(v)  # (batch_size, seq_len, d_model)

            q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
            k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
            v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

            scaled_attention, attention_weights = self.scaled_dot_product_attention(
                q, k, v, mask)

            scaled_attention = tf.transpose(scaled_attention,
                                            perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
            concat_attention = tf.reshape(scaled_attention,
                                          (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

            output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            output = self.dropout(output, training=training)

            return output, attention_weights

        def split_heads(self, x, batch_size):
            """将最后的维度分割为 (num_heads, depth)"""
            x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def scaled_dot_product_attention(self, q, k, v, mask=None):

            matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

            output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

            return output, attention_weights


    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, d_model, num_heads, dff, rate=0.1):
            super(TransformerBlock, self).__init__()

            self.mha = MultiHeadAttention(d_model, num_heads, rate)
            self.ffn = tf.keras.Sequential([
                Dense(dff, activation='relu'),
                Dense(d_model)
            ])

            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)

            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)

        def call(self, x, training=False, mask=None):

            attn_output, attn_weights = self.mha(x, x, x, mask, training=training)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)

            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)

            return out2, attn_weights


    class FlattenHead(tf.keras.layers.Layer):
        def __init__(self, n_vars, nf, target_window, head_dropout=0, **kwargs):
            super(FlattenHead, self).__init__(**kwargs)
            self.n_vars = n_vars
            self.flatten = tf.keras.layers.Flatten()
            self.linear = tf.keras.layers.Dense(target_window)
            self.dropout = tf.keras.layers.Dropout(head_dropout)

        def call(self, x, training=False):

            batch_size = tf.shape(x)[0]

            x_flat = tf.reshape(x, [batch_size * self.n_vars, -1])

            x_out = self.linear(x_flat)
            x_out = self.dropout(x_out, training=training)

            x_out = tf.reshape(x_out, [batch_size, self.n_vars, -1])

            return x_out


    class PatchTST(tf.keras.Model):
        def __init__(self, seq_len=10, pred_len=1, enc_in=1, d_model=64, n_heads=4,
                     e_layers=2, d_ff=128, dropout=0.1, activation='relu',
                     patch_len=4, stride=2, **kwargs):
            super(PatchTST, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.enc_in = enc_in
            self.d_model = d_model

            self.patch_embedding = PatchEmbedding(
                d_model, patch_len, stride, True, dropout
            )

            self.transformer_blocks = []
            for _ in range(e_layers):
                transformer_block = TransformerBlock(
                    d_model=d_model,
                    num_heads=n_heads,
                    dff=d_ff,
                    rate=dropout
                )
                self.transformer_blocks.append(transformer_block)

            self.norm_layer = LayerNormalization(epsilon=1e-6)

            self.head_nf = d_model * ((seq_len - patch_len) // stride + 1)
            self.head = FlattenHead(enc_in, self.head_nf, pred_len, dropout)

        def call(self, x, training=False):
            batch_size = tf.shape(x)[0]

            means = tf.reduce_mean(x, axis=1, keepdims=True)
            x = x - means
            stdev = tf.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev

            x_enc = tf.transpose(x, [0, 2, 1])

            enc_out, n_vars = self.patch_embedding(x_enc, training=training)

            attentions = []
            for transformer_block in self.transformer_blocks:
                enc_out, attn_weights = transformer_block(enc_out, training=training)
                attentions.append(attn_weights)


            enc_out = self.norm_layer(enc_out)


            batch_size = tf.shape(x)[0]
            enc_out = tf.reshape(enc_out, [batch_size, n_vars, -1, self.d_model])

            enc_out = tf.transpose(enc_out, [0, 1, 3, 2])

            dec_out = self.head(enc_out, training=training)

            dec_out = tf.transpose(dec_out, [0, 2, 1])

            dec_out = dec_out * stdev[:, 0:1, :]
            dec_out = dec_out + means[:, 0:1, :]

            return dec_out


    def PatchTST_time():

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


        model = PatchTST(
            seq_len=look_back,
            pred_len=1,
            enc_in=1,
            d_model=64,
            n_heads=4,
            e_layers=2,
            d_ff=4,
            dropout=0.01,
            patch_len=16,
            stride=1
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
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_data=(valX, valY),
            verbose=2
        )


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
            return 1 - (ss_res / (ss_tot))

        # 输出评估结果
        print("\n" + "=" * 50)
        print("PatchTST模型评估结果")
        print("=" * 50)

        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        pd.DataFrame({
            'y_true': testY_orig.flatten(),
            'y_pred': testPredict.flatten()
        }).to_csv('HSI_PatchTST_predictions.csv', index=False)


    if __name__ == '__main__':
        PatchTST_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")