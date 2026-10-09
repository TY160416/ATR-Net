import numpy
import time
from keras.models import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras import regularizers
import numpy as np
from A_TCN import TCN
import tensorflow as tf

np.random.seed(500)
numpy.set_printoptions(suppress=True)

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense, Dropout, Layer, Lambda, Reshape, \
    TimeDistributed

with tf.device('/GPU:0'):
    start_time = time.time()

    import tensorflow as tf
    from tensorflow.keras.layers import Layer


    class StandardKANLayer(Layer):
        """KAN layer implementation, using pure B-Spline basis functions, removing RBF approximation"""

        def __init__(self, units, grid_size=5, spline_order=3, grid_range=[-1.0, 1.0], **kwargs):
            super(StandardKANLayer, self).__init__(**kwargs)
            self.units = units
            self.grid_size = grid_size
            self.spline_order = spline_order
            self.grid_range = grid_range

        def build(self, input_shape):
            self.input_dim = input_shape[-1]

            if self.input_dim is None:
                self.input_dim = 320

            # Weights for base linear transformation
            self.base_weight = self.add_weight(
                shape=(self.input_dim, self.units),
                initializer="glorot_uniform",
                trainable=True,
                name="base_weight"
            )

            # Weights for spline functions (basis_size = grid_size + spline_order)
            self.basis_size = self.grid_size + self.spline_order
            self.spline_weight = self.add_weight(
                shape=(self.input_dim, self.units, self.basis_size),
                initializer="glorot_uniform",
                trainable=True,
                name="spline_weight"
            )

            # Build B-Spline knot grid
            # The knot range needs to be extended by spline_order positions to both left and right
            h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
            grid = tf.range(-self.spline_order, self.grid_size + self.spline_order + 1, dtype=tf.float32) * h + \
                   self.grid_range[0]
            self.grid = tf.Variable(grid, trainable=False, name="spline_grid")

            super(StandardKANLayer, self).build(input_shape)

        def b_splines(self, inputs):
            """Calculate B-Spline basis functions using the Cox-de Boor recursion formula"""
            # inputs: [batch, in_dim]
            # Expand dimensions of inputs and grid for broadcast computation
            x = tf.expand_dims(inputs, -1)  # [batch, in_dim, 1]
            grid = tf.reshape(self.grid, [1, 1, -1])  # [1, 1, grid_size + 2*spline_order + 1]

            # 0-order B-Spline (Indicator function: determines if input falls in the current grid)


    def TCN_time():
        # Read data, including Date and Close columns
        dataframe = pd.read_csv('SSE_data1.csv', usecols=['Date', 'Close'], header=0, engine='python')

        # Ensure the Date column is of datetime type
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])

        # Split the dataset according to date ranges
        train_start = '2015-01-01'
        train_end = '2021-12-31'

        val_start = '2022-01-01'
        val_end = '2023-12-31'

        test_start = '2024-01-01'
        test_end = '2025-09-29'

        # Split dataset
        train_data_raw = dataframe[(dataframe['Date'] >= train_start) & (dataframe['Date'] <= train_end)]
        val_data_raw = dataframe[(dataframe['Date'] >= val_start) & (dataframe['Date'] <= val_end)]
        test_data_raw = dataframe[(dataframe['Date'] >= test_start) & (dataframe['Date'] <= test_end)]

        print(f"Training set date range: {train_data_raw['Date'].min()} to {train_data_raw['Date'].max()}")
        print(f"Validation set date range: {val_data_raw['Date'].min()} to {val_data_raw['Date'].max()}")
        print(f"Test set date range: {test_data_raw['Date'].min()} to {test_data_raw['Date'].max()}")

        # Extract Close column data
        train_data_raw = train_data_raw['Close'].values.reshape(-1, 1)
        val_data_raw = val_data_raw['Close'].values.reshape(-1, 1)
        test_data_raw = test_data_raw['Close'].values.reshape(-1, 1)

        print(f"Original dataset split: Train {len(train_data_raw)}, Validation {len(val_data_raw)}, Test {len(test_data_raw)}")

        # Normalize the training, validation, and test sets respectively
        # Only use the training set statistics to fit the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data_raw)  # Fit using only the training set

        # Transform the respective datasets
        train_data = scaler.transform(train_data_raw)
        val_data = scaler.transform(val_data_raw)
        test_data = scaler.transform(test_data_raw)

        print(f"Normalized datasets: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")

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

        # Reshape data into 3D format [samples, timesteps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = numpy.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        print(f"Training set samples: {len(trainX)}, Validation set samples: {len(valX)}, Test set samples: {len(testX)}")

        # ========== MH-MOE Architecture, using TCN as expert networks ==========
        class TCNExpert(Layer):
            """Expert network using TCN (adapted for original MH-MOE)"""

            def __init__(self, nb_filters=64, kernel_size=3, nb_stacks=1,
                         dilations=[1, 2, 4], dropout_rate=0.01, **kwargs):
                super(TCNExpert, self).__init__(**kwargs)
                self.nb_filters = nb_filters
                self.kernel_size = kernel_size
                self.nb_stacks = nb_stacks
                self.dilations = dilations
                self.dropout_rate = dropout_rate

            def build(self, input_shape):
                input_dim = input_shape[-1]

                # Input projection layer
                self.input_projection = Dense(self.nb_filters)
                self.input_projection.build(input_shape)

                # Use the genuine TCN layer
                self.tcn = TCN(
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    nb_stacks=self.nb_stacks,
                    dilations=self.dilations,
                    padding='causal',
                    dropout_rate=self.dropout_rate,
                    return_sequences=False,
                    use_skip_connections=True,
                    use_revin=True
                )

                # Build TCN, input shape is [None, 1, nb_filters]
                self.tcn.build((None, 1, self.nb_filters))

                # Output projection back to the original dimension
                self.output_projection = Dense(input_dim)
                self.output_projection.build((None, self.nb_filters))

                super(TCNExpert, self).build(input_shape)

            def call(self, inputs, training=False):
                # inputs: [batch*seq_len, features]

                # Input projection
                projected = self.input_projection(inputs)  # [batch*seq_len, nb_filters]

                # Add time step dimension for TCN [batch*seq_len, 1, nb_filters]
                tcn_input = tf.expand_dims(projected, axis=1)

                # Process through the TCN layer
                tcn_output = self.tcn(tcn_input, training=training)  # [batch*seq_len, nb_filters]

                # Output projection back to original dimension
                output = self.output_projection(tcn_output)  # [batch*seq_len, input_dim]

                return output

        class TopkRouter(Layer):
            """Top-k router for the original MH-MOE"""

            def __init__(self, dim, num_experts, top_k, **kwargs):
                super(TopkRouter, self).__init__(**kwargs)
                self.top_k = top_k
                self.num_experts = num_experts
                self.dim = dim
                self.linear = Dense(num_experts)

            def build(self, input_shape):
                self.linear.build(input_shape)
                super(TopkRouter, self).build(input_shape)

            def call(self, x):
                # x: [batch_size, seq_len, dim] or [batch*seq_len, dim]

                # Calculate logits
                logits = self.linear(x)  # [batch_size, seq_len, num_experts] or [batch*seq_len, num_experts]

                # Retrieve top-k experts
                top_k_logits, indices = tf.math.top_k(logits, k=self.top_k)

                # Create sparse logits (set non top-k positions to negative infinity)
                batch_shape = tf.shape(logits)

                if len(logits.shape) == 3:
                    # 3D input [batch, seq_len, num_experts]
                    batch_size, seq_len, _ = batch_shape[0], batch_shape[1], batch_shape[2]

                    # Create batch and seq indices
                    batch_idx = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, seq_len, self.top_k])
                    seq_idx = tf.tile(tf.reshape(tf.range(seq_len), [1, seq_len, 1]), [batch_size, 1, self.top_k])

                    # Create indices
                    scatter_idx = tf.stack([batch_idx, seq_idx, indices], axis=-1)
                    scatter_idx = tf.reshape(scatter_idx, [-1, 3])

                else:
                    # 2D input [batch*seq_len, num_experts]
                    flat_batch = batch_shape[0]

                    # Create batch indices
                    batch_idx = tf.tile(tf.reshape(tf.range(flat_batch), [flat_batch, 1]), [1, self.top_k])
                    batch_idx = tf.expand_dims(batch_idx, -1)

                    # Create indices
                    scatter_idx = tf.concat([batch_idx, tf.expand_dims(indices, -1)], axis=-1)

                # Create sparse logits
                sparse_logits = tf.scatter_nd(
                    scatter_idx,
                    tf.reshape(top_k_logits, [-1]),
                    tf.shape(logits)
                )

                # Set non top-k positions to negative infinity
                mask = tf.equal(sparse_logits, 0)
                sparse_logits = tf.where(mask, float('-inf'), sparse_logits)

                # Calculate router output
                router_output = tf.nn.softmax(sparse_logits, axis=-1)

                return router_output, indices

        class SparseMoE(Layer):
            """Sparse MoE layer for the original MH-MOE"""

            def __init__(self, dim, num_experts, top_k, **kwargs):
                super(SparseMoE, self).__init__(**kwargs)
                self.dim = dim
                self.num_experts = num_experts
                self.top_k = top_k
                self.router = TopkRouter(dim, num_experts, top_k)
                self.experts = []

            def build(self, input_shape):
                input_dim = input_shape[-1]

                # Build the router
                self.router.build(input_shape)

                # Create TCN expert networks
                for i in range(self.num_experts):
                    expert = TCNExpert(
                        nb_filters=64,
                        kernel_size=3,
                        nb_stacks=1,
                        dilations=[1, 2, 4],
                        dropout_rate=0.01,
                        name=f'tcn_expert_{i}'
                    )
                    expert.build(input_shape)
                    self.experts.append(expert)

                super(SparseMoE, self).build(input_shape)

            def call(self, x, training=False):
                # x: [batch_size, seq_len, dim] or [batch*seq_len, dim]

                # Get routing outputs and indices
                gating_output, indices = self.router(x)

                # Initialize the final output
                final_output = tf.zeros_like(x)

                # Get input shape
                original_shape = tf.shape(x)

                # Flatten input for processing
                if len(x.shape) == 3:
                    batch_size, seq_len, dim = original_shape[0], original_shape[1], original_shape[2]
                    flat_x = tf.reshape(x, [-1, dim])
                    flat_gating_output = tf.reshape(gating_output, [-1, self.num_experts])
                    flat_indices = tf.reshape(indices, [-1, self.top_k])
                else:
                    flat_x = x
                    flat_gating_output = gating_output
                    flat_indices = indices

                # Process each expert
                for i, expert in enumerate(self.experts):
                    # Create expert mask
                    expert_mask = tf.reduce_any(tf.equal(flat_indices, i), axis=-1)

                    if tf.reduce_sum(tf.cast(expert_mask, tf.float32)) > 0:
                        # Get expert inputs
                        expert_input = tf.boolean_mask(flat_x, expert_mask)

                        # Process through expert
                        expert_output = expert(expert_input, training=training)

                        # Retrieve the corresponding gating scores
                        gating_scores = tf.boolean_mask(flat_gating_output[:, i], expert_mask)
                        gating_scores = tf.expand_dims(gating_scores, -1)

                        # Weighted outputs
                        weighted_output = expert_output * gating_scores

                        # Update final outputs
                        mask_indices = tf.where(expert_mask)
                        final_output_flat = tf.reshape(final_output, [-1, final_output.shape[-1]])

                        final_output_flat = tf.tensor_scatter_nd_add(
                            final_output_flat,
                            mask_indices,
                            weighted_output
                        )

                        # Reshape back to the original shape
                        if len(x.shape) == 3:
                            final_output = tf.reshape(final_output_flat, [batch_size, seq_len, dim])
                        else:
                            final_output = final_output_flat

                return final_output

        class Head(Layer):
            """A single self-attention head (simplified version)"""

            def __init__(self, dim, head_dim, block_size, dropout=0.1, **kwargs):
                super(Head, self).__init__(**kwargs)
                self.dim = dim
                self.head_dim = head_dim
                self.block_size = block_size
                self.dropout_rate = dropout

                # Create a lower triangular mask
                self.tril = tf.linalg.band_part(
                    tf.ones((block_size, block_size)), -1, 0
                )

                # Linear layers
                self.key = Dense(head_dim, use_bias=False)
                self.query = Dense(head_dim, use_bias=False)
                self.value = Dense(head_dim, use_bias=False)

                self.dropout = Dropout(dropout)

            def build(self, input_shape):
                self.key.build(input_shape)
                self.query.build(input_shape)
                self.value.build(input_shape)
                super(Head, self).build(input_shape)

            def call(self, x, training=False):
                # x: [batch_size, seq_len, dim]
                B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

                k = self.key(x)  # [B, T, head_dim]
                q = self.query(x)  # [B, T, head_dim]

                # Calculate attention scores
                wei = tf.matmul(q, k, transpose_b=True) * (self.head_dim ** -0.5)  # [B, T, T]

                # Apply the causal mask
                tril_mask = self.tril[:T, :T]
                mask = tf.cast(tril_mask == 0, tf.bool)
                wei = tf.where(mask, float('-inf'), wei)

                # Softmax and dropout
                wei = tf.nn.softmax(wei, axis=-1)
                wei = self.dropout(wei, training=training)

                # Weighted aggregation
                v = self.value(x)  # [B, T, head_dim]
                out = tf.matmul(wei, v)  # [B, T, head_dim]

                return out

        class MultiHeadAttention(Layer):
            """Multi-head attention (simplified version)"""

            def __init__(self, dim, heads, block_size, dropout=0.1, **kwargs):
                super(MultiHeadAttention, self).__init__(**kwargs)
                self.dim = dim
                self.heads = heads
                self.head_dim = dim // heads
                self.block_size = block_size
                self.dropout_rate = dropout

                # Create multiple attention heads
                self.head_layers = []
                for _ in range(heads):
                    head = Head(dim, self.head_dim, block_size, dropout)
                    self.head_layers.append(head)

                # Output projection
                self.proj = Dense(dim)
                self.dropout = Dropout(dropout)

            def build(self, input_shape):
                for head in self.head_layers:
                    head.build(input_shape)
                self.proj.build(input_shape)
                super(MultiHeadAttention, self).build(input_shape)

            def call(self, x, training=False):
                # Process all heads in parallel
                head_outputs = []
                for head in self.head_layers:
                    head_out = head(x, training=training)
                    head_outputs.append(head_out)

                # Concatenate outputs from all heads
                concatenated = tf.concat(head_outputs, axis=-1)  # [B, T, dim]

                # Output projection
                out = self.proj(concatenated)
                out = self.dropout(out, training=training)

                return out

        class MHMoE(Layer):
            """Multi-head MoE layer for the original MH-MOE"""

            def __init__(self, dim, heads, num_experts=4, num_layers=1, **kwargs):
                super(MHMoE, self).__init__(**kwargs)
                self.dim = dim
                self.heads = heads
                self.num_experts = num_experts
                self.num_layers = num_layers

                # Multi-head attention and MoE layers
                self.multi_head_layers = []
                self.moe_layers = []

                for i in range(num_layers):
                    # Multi-head attention layer
                    mh_layer = MultiHeadAttention(dim, heads, block_size=100, dropout=0.1)
                    self.multi_head_layers.append(mh_layer)

                    # MoE layer (each head gets its own MoE)
                    moe_layer = SparseMoE(dim // heads, num_experts, top_k=2)
                    self.moe_layers.append(moe_layer)

                # Merge layers
                self.merge_layers = []
                for i in range(num_layers):
                    merge_layer = Dense(dim)
                    self.merge_layers.append(merge_layer)

            def build(self, input_shape):
                for layer in self.multi_head_layers:
                    layer.build(input_shape)
                for layer in self.moe_layers:
                    layer.build((None, input_shape[-1] // self.heads))
                for layer in self.merge_layers:
                    layer.build(input_shape)
                super(MHMoE, self).build(input_shape)

            def call(self, x, training=False):
                # x: [batch_size, seq_len, dim]
                batch_size = tf.shape(x)[0]
                seq_len = tf.shape(x)[1]

                for i in range(self.num_layers):
                    # Multi-head attention processing
                    x_attn = self.multi_head_layers[i](x, training=training)

                    # Reshape into multi-head format
                    x_reshaped = tf.reshape(
                        x_attn,
                        [batch_size, seq_len, self.heads, self.dim // self.heads]
                    )

                    # Transpose: [batch_size, heads, seq_len, head_dim]
                    x_transposed = tf.transpose(x_reshaped, [0, 2, 1, 3])

                    # Merge batch and heads dimensions: [batch_size*heads, seq_len, head_dim]
                    batch_heads = batch_size * self.heads
                    x_flat = tf.reshape(x_transposed, [batch_heads, seq_len, self.dim // self.heads])

                    # Process through MoE layer
                    x_moe = self.moe_layers[i](x_flat, training=training)

                    # Reshape back to the original format
                    x_moe_reshaped = tf.reshape(
                        x_moe,
                        [batch_size, self.heads, seq_len, self.dim // self.heads]
                    )

                    # Transpose back: [batch_size, seq_len, heads, head_dim]
                    x_combined = tf.transpose(x_moe_reshaped, [0, 2, 1, 3])

                    # Reshape to: [batch_size, seq_len, dim]
                    x_combined = tf.reshape(x_combined, [batch_size, seq_len, self.dim])

                    # Merge layer processing
                    x = self.merge_layers[i](x_combined)

                return x

        def improved_mh_moe_module(inputs, num_experts=4, expert_units=64, num_heads=2, top_k=2):
            """
            Improved MH-MOE module, utilizing genuine TCN experts
            inputs: [batch_size, seq_len, features]
            returns: [batch_size, seq_len, expert_units]
            """
            # Obtain the input feature dimensions
            input_dim = inputs.shape[-1]

            # If input dimension is too small, directly return the projected version
            if input_dim < num_heads:
                projection = Dense(expert_units)
                return projection(inputs)

            # Use the original MH-MOE architecture
            mh_moe_output = MHMoE(
                dim=expert_units,
                heads=num_heads,
                num_experts=num_experts,
                num_layers=1
            )(inputs)

            return mh_moe_output

        # ========== Constructing the single-task model using original MH-MOE architecture ==========
        inputs = Input(shape=(look_back, 1))

        # Use the improved MH-MOE module, utilizing genuine TCN experts
        mh_moe_output = improved_mh_moe_module(
            inputs,
            num_experts=4,
            expert_units=4,  # Keep consistent with the original code
            num_heads=2,
            top_k=2
        )

        # ========== Simplified Architecture ==========
        flat_dim = look_back * 4  # seq_len * expert_units
        x = Reshape((flat_dim,))(mh_moe_output)

        # Use standard KAN layer instead of FixedKANLayer
        x = StandardKANLayer(units=16, name='kan_layer1')(x)
        x = Dropout(0.01)(x)
        x = StandardKANLayer(units=2, name='kan_layer2')(x)
        x = Dropout(0.01)(x)

        # Single-task output
        main_output = Dense(1, activation='linear', name='main_output',
                            kernel_regularizer=regularizers.l2(0.0001))(x)

        model = Model(
            inputs=[inputs],
            outputs=main_output
        )

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.995,
            epsilon=1e-8
        )

        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse']
        )

        # Train the model
        print("\nStarting training for the original MH-MOE architecture + TCN expert model...")
        history = model.fit(
            trainX, trainY,
            epochs=100,
            batch_size=32,
            shuffle=True,
            validation_data=(valX, valY),
            verbose=2
        )

        # Predictions
        print("\nMaking predictions...")
        trainPredict = model.predict(trainX)
        valPredict = model.predict(valX)
        testPredict = model.predict(testX)

        # Inverse transformations
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY_orig = scaler.inverse_transform(trainY.reshape(-1, 1))
        valPredict = scaler.inverse_transform(valPredict)
        valY_orig = scaler.inverse_transform(valY.reshape(-1, 1))
        testPredict = scaler.inverse_transform(testPredict)
        testY_orig = scaler.inverse_transform(testY.reshape(-1, 1))


        # Calculate evaluation metrics
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

        # Output evaluation results
        print("\n" + "=" * 60)
        print("Original MH-MOE with TCN Experts Model Evaluation Results")
        print("=" * 60)

        print("\n=== Test Set Evaluation ===")
        print(f"RMSE: {rmse(testPredict, testY_orig):.6f}")
        print(f"MAE: {mae(testPredict, testY_orig):.6f}")
        print(f"MAPE: {mape(testPredict, testY_orig):.6f}%")
        print(f"R²: {r_square(testPredict, testY_orig):.6f}")

        pd.DataFrame({
            'y_true': testY_orig.flatten(),
            'y_pred': testPredict.flatten()
        }).to_csv('HSI_ATR_Net_predictions.csv', index=False)


    if __name__ == '__main__':
        TCN_time()

    elapsed_time = time.time() - start_time
    print(f"\nTotal runtime: {elapsed_time:.2f} seconds")