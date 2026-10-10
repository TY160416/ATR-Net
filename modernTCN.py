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


    class ReparamLargeKernelConv1D(tf.keras.layers.Layer):
        def __init__(self, large_kernel, small_kernel, **kwargs):
            super(ReparamLargeKernelConv1D, self).__init__(**kwargs)
            self.large_kernel = large_kernel
            self.small_kernel = small_kernel

        def build(self, input_shape):
            self.large_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.large_kernel, 1),
                padding='same',
                use_bias=False
            )
            self.large_bn = tf.keras.layers.BatchNormalization()

            if self.small_kernel is not None and self.small_kernel <= self.large_kernel:
                self.small_conv = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=(self.small_kernel, 1),
                    padding='same',
                    use_bias=False
                )
                self.small_bn = tf.keras.layers.BatchNormalization()
            else:
                self.small_conv = None

            super(ReparamLargeKernelConv1D, self).build(input_shape)

        def call(self, x, training=False):
            # x: (batch, seq_len, channels) -> 升维成 (batch, seq_len, 1, channels) 用 2D depthwise conv 代替
            x_exp = tf.expand_dims(x, axis=2)

            out = self.large_bn(self.large_conv(x_exp), training=training)
            if self.small_conv is not None:
                out += self.small_bn(self.small_conv(x_exp), training=training)

            out = tf.squeeze(out, axis=2)  # 降回 (batch, seq_len, channels)
            return out


    class ModernTCNBlock(tf.keras.layers.Layer):
        def __init__(self, dmodel, dff, large_kernel, small_kernel, drop=0.1, **kwargs):
            super(ModernTCNBlock, self).__init__(**kwargs)
            self.dw = ReparamLargeKernelConv1D(large_kernel, small_kernel)
            self.norm = tf.keras.layers.BatchNormalization()

            self.ffn1pw1 = tf.keras.layers.Conv1D(filters=dff, kernel_size=1)
            self.ffn1act = tf.keras.layers.Activation('gelu')
            self.ffn1pw2 = tf.keras.layers.Conv1D(filters=dmodel, kernel_size=1)
            self.ffn1drop1 = tf.keras.layers.Dropout(drop)
            self.ffn1drop2 = tf.keras.layers.Dropout(drop)

            self.ffn2pw1 = tf.keras.layers.Conv1D(filters=dff, kernel_size=1)
            self.ffn2act = tf.keras.layers.Activation('gelu')
            self.ffn2pw2 = tf.keras.layers.Conv1D(filters=dmodel, kernel_size=1)
            self.ffn2drop1 = tf.keras.layers.Dropout(drop)
            self.ffn2drop2 = tf.keras.layers.Dropout(drop)

        def call(self, x, training=False):
            res = x
            x = self.dw(x, training=training)
            x = self.norm(x, training=training)

            x1 = self.ffn1drop1(self.ffn1pw1(x), training=training)
            x1 = self.ffn1act(x1)
            x1 = self.ffn1drop2(self.ffn1pw2(x1), training=training)

            x2 = self.ffn2drop1(self.ffn2pw1(x1), training=training)
            x2 = self.ffn2act(x2)
            x2 = self.ffn2drop2(self.ffn2pw2(x2), training=training)

            return res + x2


    class ModernTCN(tf.keras.Model):
        def __init__(self, seq_len=10, pred_len=1, enc_in=1,
                     dims=[32, 64], large_sizes=[5, 5], small_sizes=[3, 3],
                     num_blocks=[1, 1], ffn_ratio=2,
                     patch_size=2, patch_stride=1, dropout=0.1, **kwargs):
            super(ModernTCN, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len

            self.stem = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=dims[0], kernel_size=patch_size,
                                       strides=patch_stride, padding='same'),
                tf.keras.layers.BatchNormalization()
            ])

            self.stages = []
            for i in range(len(dims)):
                stage_blocks = tf.keras.Sequential([
                    ModernTCNBlock(dmodel=dims[i], dff=dims[i] * ffn_ratio,
                                   large_kernel=large_sizes[i], small_kernel=small_sizes[i],
                                   drop=dropout)
                    for _ in range(num_blocks[i])
                ])
                self.stages.append(stage_blocks)

                if i < len(dims) - 1:
                    downsample = tf.keras.Sequential([
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Conv1D(filters=dims[i + 1], kernel_size=1, strides=1)
                    ])
                    self.stages.append(downsample)

            self.flatten = tf.keras.layers.Flatten()
            self.head_drop = tf.keras.layers.Dropout(dropout)
            self.head = tf.keras.layers.Dense(pred_len)

        def call(self, x, training=False):
            means = tf.reduce_mean(x, axis=1, keepdims=True)
            x = x - means
            stdev = tf.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev

            x = self.stem(x, training=training)
            for layer in self.stages:
                x = layer(x, training=training)

            x = self.flatten(x)
            x = self.head_drop(x, training=training)
            x = self.head(x)

            x = tf.reshape(x, [-1, self.pred_len, 1])

            x = x * stdev[:, 0:1, :]
            x = x + means[:, 0:1, :]

            return x


    def ModernTCN_time():
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

        model = ModernTCN(
            seq_len=look_back,
            pred_len=1,
            enc_in=1,
            dims=[32, 64],
            large_sizes=[5, 5],
            small_sizes=[3, 3],
            num_blocks=[1, 1],
            ffn_ratio=2,
            patch_size=2,
            patch_stride=1,
            dropout=0.01
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
        print("ModernTCN 模型评估结果")
        print("=" * 50)

        print("\n=== 测试集评估 ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

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

        # 保存为 CSV
        output_filename = 'HSI_modernTCN_predictions.csv'
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已成功保存至本地文件: {output_filename}")
        # --------------------------------------------------------------


    if __name__ == '__main__':
        ModernTCN_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")