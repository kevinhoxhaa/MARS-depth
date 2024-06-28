import numpy as np
import keras

from keras.api.layers import Dense
from keras.api.layers import BatchNormalization
from keras.api.layers import Conv1D
from keras.api.layers import GlobalMaxPooling1D
from keras.api.layers import Reshape
from keras.api.layers import Dot
import matplotlib.pyplot as plt

'''
This code was inspired by the following sources:
- https://keras.io/examples/vision/pointnet/#build-a-model
- https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py
'''


# Define the PointNet model
def pointnet_model(num_points, num_features, num_classes):
    input_points = keras.layers.Input(shape=(num_points, num_features))

    # PointNet Classification Network
    transformed_inputs = T_net(
        input_points, num_features=num_features, name="input_transformation_block"
    )
    features_64 = conv_layer(transformed_inputs, filters=64, name="features_64")

    features_128 = conv_layer(features_64, filters=128, name="features_128")
    features_512 = conv_layer(features_128, filters=512, name="features_512")

    global_features = GlobalMaxPooling1D(name="global_features")(features_512)

    # Fully connected layers
    fc_256 = dense_layer(global_features, filters=256, name="fc_256")
    fc_128 = dense_layer(fc_256, filters=128, name="fc_128")
    fc_64 = dense_layer(fc_128, filters=64, name="fc_64")
    outputs = Dense(num_classes, activation="linear", name="outputs")(fc_64)

    model = keras.Model(input_points, outputs)
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5)
    # compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', keras.metrics.RootMeanSquaredError()])
    return model


def conv_layer(x, filters, name):
    x = Conv1D(filters, kernel_size=3, activation='relu', padding="valid", name=f"{name}_conv")(x)
    return BatchNormalization(name=f"{name}_batch_norm", momentum=0.95)(x)


def dense_layer(x, filters, name):
    x = Dense(filters, name=f"{name}_dense", activation='relu')(x)
    return BatchNormalization(name=f"{name}_batch_norm", momentum=0.95)(x)


def T_net(inputs, num_features, name):
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.
    """
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())

    x = conv_layer(inputs, filters=64, name=f"{name}_1")
    x = conv_layer(x, filters=128, name=f"{name}_2")
    x = conv_layer(x, filters=512, name=f"{name}_3")
    x = GlobalMaxPooling1D(name="T_net_global_features")(x)
    x = dense_layer(x, filters=256, name=f"{name}_1_1")
    x = dense_layer(x, filters=128, name=f"{name}_2_1")
    x = dense_layer(x, filters=64, name=f"{name}_3_1")
    transformed_features = Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        name=f"{name}_final",
    )(x)
    transformed_features = Reshape((num_features, num_features))(
        transformed_features
    )
    return Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


def plot_barchart(mars_scores, pointnet_scores):
    """
    Plot the bar chart of the MAEs of the Mars and PointNet models
    """
    # Plot the bar chart of the MAEs and RMSEs of the Mars and PointNet models
    plt.figure(figsize=(10, 5))
    x = np.arange(len(mars_scores))
    plt.bar(x - 0.175, mars_scores, 0.35)
    plt.bar(x + 0.175, pointnet_scores, 0.35)
    plt.xticks(x, ['MAE', 'RMSE'])
    plt.legend(['Mars', 'PointNet'])

    plt.title('MARS with CNN vs PointNet Comparison')
    plt.xlabel('Error Metrics')
    plt.ylabel('Distance (cm)')
    plt.savefig('model/plots/model_comparison.png')
    plt.show()
    plt.close()


def plot_all_barchart():
    """
    Plot the bar chart for multiple models.

    Parameters:
    - model_scores: list of lists, where each inner list contains scores for a model.
    - model_labels: list of strings, names of the models.

    Each inner list in model_scores should have the same length, corresponding to the number of metrics (e.g., MAE, RMSE).
    """
    model_scores = [
        np.load('model/Accuracy/normal_MARS/MARS_accuracy.npy')[-1, 6:],  # Model 1 scores
        np.load('model/Accuracy/optimal_MARS/MARS_accuracy.npy')[-1, 6:],  # Model 3 scores
        np.load('model/Accuracy/pointnet/PointNet_accuracy.npy')[-1, 6:]  # Model 4 scores
    ]
    model_labels = ['Default MARS', 'Optimal MARS', 'PointNet']

    num_metrics = len(model_scores[0])  # Assuming each model has the same number of scores
    num_models = len(model_scores)

    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # A pleasant and distinct set of colors

    plt.figure(figsize=(10, 5))
    x = np.arange(num_metrics)
    bar_width = 0.8 / num_models  # Divide the bar width equally among the models

    for i, scores in enumerate(model_scores):
        plt.bar(x + i * bar_width, scores, bar_width, label=model_labels[i], color=colors[i])

    plt.xticks(x + bar_width * (num_models - 1) / 2, ['MAE', 'RMSE'])
    plt.legend()

    plt.title('Model Comparison Across Different Metrics')
    plt.xlabel('Error Metrics')
    plt.ylabel('Distance (cm)')
    plt.savefig('model/plots/model_comparison_all.png')
    plt.show()
    plt.close()

