import numpy as np
import keras

from keras.api.layers import Dense
from keras.api.layers import BatchNormalization
from keras.api.layers import Activation
from keras.api.layers import Conv1D
from keras.api.layers import GlobalMaxPooling1D
from keras.api.layers import Reshape
from keras.api.layers import Dot
from keras.api.layers import MaxPool1D
from keras.api.layers import Concatenate

from orthogonal_regularizer import OrthogonalRegularizer
'''
This code was inspired by the following sources:
- https://keras.io/examples/vision/pointnet/#build-a-model
- https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet_segmentation.py
'''

def pointnet_model(num_points, num_classes):
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network
    transformed_inputs = transformation_net(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_layer(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_layer(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_layer(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_net(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_layer(transformed_features, filters=512, name="features_512")
    features_2048 = conv_layer(features_512, filters=2048, name="pre_maxpool_block")
    global_features = MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = keras.ops.tile(global_features, [1, num_points, 1])

    # PointNet Segmentation Input
    segmentation_input = Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_layer(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)

    model = keras.Model(input_points, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    # compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', keras.metrics.RootMeanSquaredError()])
    return model


def conv_layer(x, filters, name):
    x = Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = BatchNormalization(name=f"{name}_batch_norm")(x)
    return Activation("relu", name=f"{name}_relu")(x)


def dense_layer(x, filters, name):
    x = Dense(filters, name=f"{name}_dense")(x)
    x = BatchNormalization(name=f"{name}_batch_norm")(x)
    return Activation("relu", name=f"{name}_relu")(x)


def transformation_net(inputs, num_features, name):
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_layer(inputs, filters=64, name=f"{name}_1")
    x = conv_layer(x, filters=128, name=f"{name}_2")
    x = conv_layer(x, filters=1024, name=f"{name}_3")
    x = GlobalMaxPooling1D()(x)
    x = dense_layer(x, filters=512, name=f"{name}_1_1")
    x = dense_layer(x, filters=256, name=f"{name}_2_1")
    transformed_features = Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)
    transformed_features = Reshape((num_features, num_features))(
        transformed_features
    )
    return Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])