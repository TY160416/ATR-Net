import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def load_sse_data(
    csv_path="SSE_data1.csv",
    look_back=10,
    train_start="2015-01-01",
    train_end="2021-12-31",
    val_start="2022-01-01",
    val_end="2023-12-31",
    test_start="2024-01-01",
    test_end="2025-09-29",
):
    dataframe = pd.read_csv(csv_path, usecols=["Date", "Close"], header=0, engine="python")
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])

    train_df = dataframe[(dataframe["Date"] >= train_start) & (dataframe["Date"] <= train_end)]
    val_df = dataframe[(dataframe["Date"] >= val_start) & (dataframe["Date"] <= val_end)]
    test_df = dataframe[(dataframe["Date"] >= test_start) & (dataframe["Date"] <= test_end)]

    print(f"Training set date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"Validation set date range: {val_df['Date'].min()} to {val_df['Date'].max()}")
    print(f"Test set date range: {test_df['Date'].min()} to {test_df['Date'].max()}")

    train_raw = train_df["Close"].values.reshape(-1, 1)
    val_raw = val_df["Close"].values.reshape(-1, 1)
    test_raw = test_df["Close"].values.reshape(-1, 1)

    print(
        f"Original dataset split: Train {len(train_raw)}, Validation {len(val_raw)}, Test {len(test_raw)}"
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_raw)

    train_data = scaler.transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    print(
        f"Normalized datasets: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}"
    )

    trainX, trainY = create_dataset(train_data, look_back)
    valX, valY = create_dataset(val_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    print(f"Training set samples: {len(trainX)}, Validation set samples: {len(valX)}, Test set samples: {len(testX)}")

    return scaler, (trainX, trainY), (valX, valY), (testX, testY)