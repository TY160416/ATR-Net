import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.layers import Input, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from data_loader import load_sse_data
from layers.standard_kan import StandardKANLayer
from models.mhmoe_tcn import improved_mh_moe_module


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def mape(predictions, targets):
    return np.mean(np.abs((predictions - targets) / targets)) * 100

def r_square(predictions, targets):
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    return 1 - (ss_res / ss_tot)


def TCN_time():
    np.random.seed(500)
    tf.random.set_seed(500)

    look_back = 10
    scaler, (trainX, trainY), (valX, valY), (testX, testY) = load_sse_data(
        csv_path="SSE_data1.csv",
        look_back=look_back,
    )

    inputs = Input(shape=(look_back, 1))

    mh_moe_output = improved_mh_moe_module(
        inputs,
        num_experts=4,
        expert_units=4,
        num_heads=2,
        top_k=2,
    )

    flat_dim = look_back * 4
    x = Reshape((flat_dim,))(mh_moe_output)

    x = StandardKANLayer(units=16, name="kan_layer1")(x)
    x = Dropout(0.01)(x)
    x = StandardKANLayer(units=2, name="kan_layer2")(x)
    x = Dropout(0.01)(x)

    main_output = Dense(
        1,
        activation="linear",
        name="main_output",
        kernel_regularizer=regularizers.l2(0.0001),
    )(x)

    model = Model(inputs=[inputs], outputs=main_output)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.995,
        epsilon=1e-8,
    )

    model.compile(
        loss="mse",
        optimizer=optimizer,
        metrics=["mae", "mse"],
    )

    print("\nStarting training for the original MH-MOE architecture + TCN expert model...")
    model.fit(
        trainX,
        trainY,
        epochs=100,
        batch_size=32,
        shuffle=True,
        validation_data=(valX, valY),
        verbose=2,
    )

    print("\nMaking predictions...")
    trainPredict = model.predict(trainX)
    valPredict = model.predict(valX)
    testPredict = model.predict(testX)

    # inverse transform
    trainPredict = scaler.inverse_transform(trainPredict)
    valPredict = scaler.inverse_transform(valPredict)
    testPredict = scaler.inverse_transform(testPredict)

    trainY_orig = scaler.inverse_transform(trainY.reshape(-1, 1))
    valY_orig = scaler.inverse_transform(valY.reshape(-1, 1))
    testY_orig = scaler.inverse_transform(testY.reshape(-1, 1))

    print("\n" + "=" * 60)
    print("Original MH-MOE with TCN Experts Model Evaluation Results")
    print("=" * 60)

    print("\n=== Test Set Evaluation ===")
    print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
    print(f"MAE: {mae(testPredict, testY_orig):.6f}")
    print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
    print(f"R²: {r_square(testPredict, testY_orig):.6f}")

    pd.DataFrame({"y_true": testY_orig.flatten(), "y_pred": testPredict.flatten()}).to_csv(
        "HSI_ATR_Net_predictions.csv", index=False
    )


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()
    with tf.device("/GPU:0"):
        TCN_time()
    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")