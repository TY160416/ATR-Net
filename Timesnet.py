import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import tensorflow as tf

np.random.seed(500)
np.set_printoptions(suppress=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with tf.device('/GPU:0'):
    start_time = time.time()


    class InceptionBlockV1(tf.keras.layers.Layer):
        """参数高效的 2D Inception 模块"""

        def __init__(self, out_channels, num_kernels=3, **kwargs):
            super(InceptionBlockV1, self).__init__(**kwargs)
            self.out_channels = out_channels
            self.num_kernels = num_kernels

        def build(self, input_shape):
            kernels = [1, 3, 5, 7][:self.num_kernels]
            self.convs = []
            for k in kernels:
                self.convs.append(tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        filters=self.out_channels,
                        kernel_size=(k, k),
                        padding='same',
                        activation='linear',
                        use_bias=False
                    )
                ]))
            super(InceptionBlockV1, self).build(input_shape)

        def call(self, x, training=False):
            res = []
            for conv in self.convs:
                res.append(conv(x, training=training))
            # 聚合多个卷积核结果
            return tf.reduce_mean(tf.stack(res, axis=0), axis=0)


    class TimesBlock(tf.keras.layers.Layer):

        def __init__(self, seq_len, pred_len, k, d_model, d_ff, num_kernels, **kwargs):
            super(TimesBlock, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.k = k
            self.d_model = d_model

            self.conv1 = InceptionBlockV1(d_ff, num_kernels)
            self.gelu = tf.keras.layers.Activation('gelu')
            self.conv2 = InceptionBlockV1(d_model, num_kernels)

        def call(self, x, training=False):
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            C = tf.shape(x)[2]

            x_t = tf.transpose(x, [0, 2, 1])
            xf = tf.signal.rfft(x_t)
            amp = tf.abs(xf)
            amp = tf.transpose(amp, [0, 2, 1])  # [B, T//2+1, C]

            freq_list = tf.reduce_mean(amp, axis=[0, 2])
            freq_list = tf.tensor_scatter_nd_update(freq_list, [[0]], [0.0])  # 将常数项(0频)置零
            _, top_indices = tf.math.top_k(freq_list, k=self.k)

            amp_mean = tf.reduce_mean(amp, axis=-1)

            res = []
            period_weights = []

            for i in range(self.k):
                top_index = tf.cast(top_indices[i], tf.int32)
                period = tf.cast(T // tf.math.maximum(top_index, 1), tf.int32)

                # 填充以满足 2D reshape 要求
                pad_len = tf.math.floormod(T, period)
                pad_len = tf.where(pad_len == 0, 0, period - pad_len)
                paddings = tf.stack([[0, 0], [0, pad_len], [0, 0]])
                out = tf.pad(x, paddings)

                # 转化为 2D 表示
                shape_in = tf.stack([B, -1, period, C])
                out = tf.reshape(out, shape_in)

                # 2D 卷积提取变异性
                out = self.conv1(out, training=training)
                out = self.gelu(out)
                out = self.conv2(out, training=training)

                # 转回 1D
                shape_out = tf.stack([B, -1, self.d_model])
                out = tf.reshape(out, shape_out)
                out = out[:, :T, :]  # 截断填充部分
                res.append(out)

                weight = amp_mean[:, top_index]
                period_weights.append(weight)

            res = tf.stack(res, axis=-1)  # [B, T, d_model, k]

            period_weights = tf.stack(period_weights, axis=-1)
            period_weights = tf.nn.softmax(period_weights, axis=-1)
            period_weights = tf.reshape(period_weights, [B, 1, 1, self.k])

            res = tf.reduce_sum(res * period_weights, axis=-1)

            return res + x


    class TimesNet(tf.keras.Model):

        def __init__(self, seq_len=10, pred_len=1, enc_in=1,
                     d_model=32, d_ff=64, e_layers=2, top_k=2,
                     num_kernels=3, dropout=0.1, **kwargs):
            super(TimesNet, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len

            # 简易特征嵌入与预测时长拓展
            self.enc_embedding = tf.keras.layers.Dense(d_model)
            self.predict_linear = tf.keras.layers.Dense(seq_len + pred_len)

            # 堆叠多层 TimesBlock
            self.times_blocks = []
            for _ in range(e_layers):
                self.times_blocks.append(TimesBlock(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    k=top_k,
                    d_model=d_model,
                    d_ff=d_ff,
                    num_kernels=num_kernels
                ))
                self.times_blocks.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

            self.projection = tf.keras.layers.Dense(enc_in)

        def call(self, x, training=False):
            means = tf.reduce_mean(x, axis=1, keepdims=True)
            x = x - means
            stdev = tf.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev

            enc_out = self.enc_embedding(x)
            enc_out = tf.transpose(enc_out, [0, 2, 1])
            enc_out = self.predict_linear(enc_out)
            enc_out = tf.transpose(enc_out, [0, 2, 1])


            for i in range(len(self.times_blocks) // 2):
                block = self.times_blocks[2 * i]
                norm = self.times_blocks[2 * i + 1]
                enc_out = block(enc_out, training=training)
                enc_out = norm(enc_out)

            dec_out = self.projection(enc_out)

            dec_out = dec_out * stdev[:, 0:1, :]
            dec_out = dec_out + means[:, 0:1, :]

            return dec_out[:, -self.pred_len:, :]


    def TimesNet_time():
        # 数据加载及预处理
        dataframe = pd.read_csv('HSI_data1.csv', usecols=['Date', 'Close'], header=0, engine='python')
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])

        train_start, train_end = '2015-01-01', '2021-12-31'
        val_start, val_end = '2022-01-01', '2023-12-31'
        test_start, test_end = '2024-01-01', '2025-09-29'

        train_data_raw = dataframe[(dataframe['Date'] >= train_start) & (dataframe['Date'] <= train_end)]
        val_data_raw = dataframe[(dataframe['Date'] >= val_start) & (dataframe['Date'] <= val_end)]
        test_data_raw = dataframe[(dataframe['Date'] >= test_start) & (dataframe['Date'] <= test_end)]

        train_data_values = train_data_raw['Close'].values.reshape(-1, 1)
        val_data_values = val_data_raw['Close'].values.reshape(-1, 1)
        test_data_values = test_data_raw['Close'].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_values)

        train_data = scaler.transform(train_data_values)
        val_data = scaler.transform(val_data_values)
        test_data = scaler.transform(test_data_values)

        def create_dataset(dataset, look_back):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back)]
                dataX.append(a)
                dataY.append(dataset[i + look_back])
            return np.array(dataX), np.array(dataY)

        look_back = 10
        trainX, trainY = create_dataset(train_data, look_back)
        valX, valY = create_dataset(val_data, look_back)
        testX, testY = create_dataset(test_data, look_back)

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        model = TimesNet(
            seq_len=look_back,
            pred_len=1,
            enc_in=1,
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=2,
            num_kernels=3,
            dropout=0.1
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
            return np.sqrt(((predictions - targets) ** 2).mean())

        def mae(predictions, targets):
            return np.mean(np.abs(predictions - targets))

        def mape(predictions, targets):
            return np.mean(np.abs((predictions - targets) / targets)) * 100

        def r_square(predictions, targets):
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return 1 - (ss_res / (ss_tot))

        print("\n" + "=" * 50)
        print("TimesNet 模型评估结果")
        print("=" * 50)

        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        # ------------------ 保存预测结果至 CSV ------------------
        test_dates = test_data_raw['Date'].iloc[look_back:].dt.strftime('%Y/%m/%d').values

        actual_vals = testY_orig.flatten()
        predicted_vals = testPredict.flatten()
        errors = actual_vals - predicted_vals

        results_df = pd.DataFrame({
            'Data': test_dates,
            'Actual': actual_vals,
            'Predicted': predicted_vals,
            'Error': errors
        })

        output_filename = 'HSI_TimesNet_predictions.csv'
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已成功保存至本地文件: {output_filename}")


    if __name__ == '__main__':
        TimesNet_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")