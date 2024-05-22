import numpy as np
import keras

from keras.api.layers import Dense
from keras.api.layers import BatchNormalization
from keras.api.layers import Conv1D
from keras.api.layers import GlobalMaxPooling1D
from keras.api.layers import Reshape
from keras.api.layers import Dot

from orthogonal_regularizer import OrthogonalRegularizer
'''
This code was inspired by the following sources:
- https://keras.io/examples/vision/pointnet/#build-a-model
- https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet_segmentation.py
'''

def pointnet_model(num_points, num_features, num_classes):
    input_points = keras.layers.Input(shape=(num_points, num_features))

    # PointNet Classification Network
    transformed_inputs = T_net(
        input_points, num_features=num_features, name="input_transformation_block"
    )
    features_64 = conv_layer(transformed_inputs, filters=64, name="features_64")

    features_128 = conv_layer(features_64, filters=128, name="features_128")
    features_1024 = conv_layer(features_128, filters=1024, name="features_1024")

    global_features = GlobalMaxPooling1D(name="global_features")(features_1024)


    # Fully connected layers
    fc_512 = dense_layer(global_features, filters=512, name="fc_512")
    fc_256 = dense_layer(fc_512, filters=256, name="fc_256")
    outputs = Dense(num_classes, activation="linear", name="outputs")(fc_256)

    model = keras.Model(input_points, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    # compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', keras.metrics.RootMeanSquaredError()])
    return model


def conv_layer(x, filters, name):
    x = Conv1D(filters, kernel_size=3, activation='relu', padding="same", name=f"{name}_conv")(x)
    return BatchNormalization(name=f"{name}_batch_norm")(x)


def dense_layer(x, filters, name):
    x = Dense(filters, name=f"{name}_dense", activation='relu')(x)
    return BatchNormalization(name=f"{name}_batch_norm")(x)


def T_net(inputs, num_features, name):
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_layer(inputs, filters=64, name=f"{name}_1")
    x = conv_layer(x, filters=128, name=f"{name}_2")
    x = conv_layer(x, filters=1024, name=f"{name}_3")
    x = GlobalMaxPooling1D(name="T_net_global_features")(x)
    x = dense_layer(x, filters=512, name=f"{name}_1_1")
    x = dense_layer(x, filters=256, name=f"{name}_2_1")
    transformed_features = Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        # activity_regularizer=reg,
        name=f"{name}_final",
    )(x)
    transformed_features = Reshape((num_features, num_features))(
        transformed_features
    )
    return Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])