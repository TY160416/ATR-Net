import math
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


    class MambaBlock(tf.keras.layers.Layer):

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
            super(MambaBlock, self).__init__(**kwargs)
            self.d_model = d_model
            self.d_state = d_state
            self.d_conv = d_conv
            self.expand = expand
            self.d_inner = int(expand * d_model)
            self.dt_rank = max(1, math.ceil(d_model / 16))

        def build(self, input_shape):

            self.in_proj = tf.keras.layers.Dense(self.d_inner * 2, use_bias=False)
            self.conv1d = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.d_conv, 1),
                padding='valid',
                use_bias=True
            )

            self.x_proj = tf.keras.layers.Dense(self.dt_rank + self.d_state * 2, use_bias=False)
            self.dt_proj = tf.keras.layers.Dense(self.d_inner, use_bias=True)

            A_init = tf.repeat(tf.range(1, self.d_state + 1, dtype=tf.float32)[None, :], self.d_inner, axis=0)
            self.A_log = tf.Variable(tf.math.log(A_init), trainable=True, name="A_log")
            self.D = tf.Variable(tf.ones(self.d_inner), trainable=True, name="D")

            self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=False)
            super(MambaBlock, self).build(input_shape)

        def call(self, x, training=False):
            # x shape: [B, L, d_model]
            xz = self.in_proj(x)
            x_conv, z = tf.split(xz, 2, axis=-1)

            pad_len = self.d_conv - 1
            x_conv_padded = tf.pad(x_conv, [[0, 0], [pad_len, 0], [0, 0]])
            x_conv_exp = tf.expand_dims(x_conv_padded, axis=2)
            x_conv = self.conv1d(x_conv_exp)
            x_conv = tf.squeeze(x_conv, axis=2)
            x_conv = tf.nn.silu(x_conv)

            # 投影生成选择性参数: dt, B, C
            x_dbl = self.x_proj(x_conv)
            dt, B_mat, C_mat = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)

            dt = self.dt_proj(dt)
            dt = tf.nn.softplus(dt)

            A = -tf.exp(self.A_log)  # [d_inner, d_state]

            # --- 离散化与选择性扫描 (Selective Scan) ---
            # dA = exp(dt * A)
            dA = tf.exp(tf.einsum('bld,dn->bldn', dt, A))
            # dB_u = dt * B * x
            dB_u = tf.einsum('bld,bln,bld->bldn', dt, B_mat, x_conv)

            # 转置为 [L, B, d_inner, d_state] 适应 tf.scan
            dA_t = tf.transpose(dA, [1, 0, 2, 3])
            dB_u_t = tf.transpose(dB_u, [1, 0, 2, 3])

            def scan_step(h_prev, inputs):
                dA_t_step, dB_u_t_step = inputs
                return dA_t_step * h_prev + dB_u_t_step

            h = tf.scan(scan_step, (dA_t, dB_u_t), initializer=tf.zeros_like(dA_t[0]))

            # 结果转置回 [B, L, d_inner, d_state]
            h = tf.transpose(h, [1, 0, 2, 3])

            # y = C * h + D * x
            y = tf.einsum('bldn,bln->bld', h, C_mat)
            y = y + x_conv * self.D

            # 门控分支
            y = y * tf.nn.silu(z)
            out = self.out_proj(y)
            return out


    class MambaModel(tf.keras.Model):

        def __init__(self, seq_len=10, pred_len=1, enc_in=1,
                     d_model=32, d_ff=16, d_conv=4, expand=2, dropout=0.1, **kwargs):
            super(MambaModel, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.pred_len = pred_len

            # 特征嵌入
            self.enc_embedding = tf.keras.layers.Dense(d_model)

            # Mamba 核心层
            self.mamba = MambaBlock(
                d_model=d_model,
                d_state=d_ff,
                d_conv=d_conv,
                expand=expand
            )

            # 预测输出层
            self.out_layer = tf.keras.layers.Dense(enc_in, use_bias=False)

        def call(self, x, training=False):
            means = tf.reduce_mean(x, axis=1, keepdims=True)
            x = x - means
            stdev = tf.sqrt(tf.math.reduce_variance(x, axis=1, keepdims=True) + 1e-5)
            x = x / stdev

            x_emb = self.enc_embedding(x)
            x_mamba = self.mamba(x_emb, training=training)

            x_out = self.out_layer(x_mamba)

            x_out = x_out[:, -self.pred_len:, :]

            x_out = x_out * stdev[:, 0:1, :]
            x_out = x_out + means[:, 0:1, :]

            return x_out


    def Mamba_time():
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

        # 实例化 Mamba 模型
        model = MambaModel(
            seq_len=look_back,
            pred_len=1,
            enc_in=1,
            d_model=32,
            d_ff=16,  # 等价于 Mamba 中的 d_state
            d_conv=4,
            expand=2,
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

        # 通过传入占位符建立模型图
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
        print("Mamba 模型评估结果")
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

        output_filename = 'HSI_Mamba_predictions.csv'
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n预测结果已成功保存至本地文件: {output_filename}")


    if __name__ == '__main__':
        Mamba_time()

    elapsed_time = time.time() - start_time
    print(f"\n总运行时间: {elapsed_time:.2f} 秒")