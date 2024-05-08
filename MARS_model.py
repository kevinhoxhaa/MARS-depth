# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:59:44 2021

@author: xxx
"""

"""
import all the necessary packages
Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""
import matplotlib.pyplot as plt
import numpy as np
import keras
import time
from sklearn import metrics

from keras.api.optimizers import Adam
from keras.api.models import Model
from keras.api.layers import Dense
from keras.api.layers import Input
from keras.api.layers import Flatten
from keras.api.layers import Conv2D
from keras.api.layers import BatchNormalization
from keras.api.layers import Dropout
from keras.api.layers import Activation
from keras.api.layers import Conv1D
from keras.api.layers import GlobalMaxPooling1D
from keras.api.layers import Reshape
from keras.api.layers import Dot
from keras.api.layers import MaxPool1D
from keras.api.layers import Concatenate

# set the directory
import os
path = os.getcwd()
os.chdir(path)


# Load the feature and labels, 24066, 8033, and 7984 frames for train, validate, and test
featuremap_train = np.load('feature/featuremap_train.npy')
featuremap_validate = np.load('feature/featuremap_validate.npy')
featuremap_test = np.load('feature/featuremap_test.npy')

labels_train = np.load('feature/labels_train.npy')
labels_validate = np.load('feature/labels_validate.npy')
labels_test = np.load('feature/labels_test.npy')

# Initialize the result array
paper_result_list = []


# define batch size and epochs
batch_size = 128
epochs = 150

# If you are testing the convolutional layers, this value is 'convolutional',
# if you are testing the dense layers, this value is 'dense'
validation_type = 'dense'

# If you are using the PointNet architecture, set the architecture_type to 'POINTNET'
# If you are using the MARS architecture, set the architecture_type to 'MARS'
architecture_type = 'MARS'


# define the model
def define_CNN(in_shape, n_keypoints, num_conv_layers, num_dense_layers):
    """
    Define the CNN model for keypoint detection with the given number of convolutional and dense layers.
    """

    # Input layer
    in_one = Input(shape=in_shape)
    layer = in_one

    # Create the convolutional layers
    for i in range(num_conv_layers):
        layer = Conv2D(16 * (2 ** i), kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(layer)
        layer = Dropout(0.3)(layer)
        layer = BatchNormalization(momentum=0.95)(layer)

    # Flatten the output of the convolutional layers
    dense_layer = Flatten()(layer)

    # Create the dense layers
    for i in range(num_dense_layers):
        dense_layer = Dense(int(512 * ((1/2) ** i)), activation='relu')(dense_layer)
        dense_layer = BatchNormalization(momentum=0.95)(dense_layer)
        dense_layer = Dropout(0.4)(dense_layer)

    # Output layer
    out_layer = Dense(n_keypoints, activation='linear')(dense_layer)

    # Build the model with the input and output layers
    model = Model(in_one, out_layer)
    opt = Adam(learning_rate=0.001, beta_1=0.5)

    # compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', keras.metrics.RootMeanSquaredError()])
    return model


# define the number of layers to test
num_conv_layers = [1, 2, 3, 4]
# define the number of dense layers to test
num_dense_layers = [1, 2, 3, 4]

avg_mae_list = []
avg_val_mae_list = []
avg_loss_list = []
avg_val_loss_list = []
times = []

# loop through the number of layers to test
# if validation_type is 'layer', test the number of convolutional layers
# else, test the number of dense layers
num_layers = num_conv_layers if validation_type == 'convolutional' else num_dense_layers
for n in num_layers:
    if validation_type == 'convolutional':
        print('Number of Convolutional Layers:', n)
    else:
        print('Number of Dense Layers:', n)

    conv_times = []
    avg_mae = []
    avg_val_mae = []
    avg_loss = []
    avg_val_loss = []

    # Repeat i iteration to get the average result
    for i in range(10):
        # instantiate the model
        if architecture_type == 'MARS':
            keypoint_model = define_CNN(featuremap_train[0].shape, 57, n, 1)
        # elif architecture_type == 'POINTNET':
        #     keypoint_model =
        # initial maximum error
        score_min = 10
        start_time = time.time()
        history = keypoint_model.fit(featuremap_train, labels_train,
                                     batch_size=batch_size, epochs=epochs, verbose=1,
                                     validation_data=(featuremap_validate, labels_validate))
        conv_times.append(time.time() - start_time)
        print('Time taken for training:', conv_times[-1])
        # save and print the metrics
        score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
        print('train MAPE = ', score_train[3])
        score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
        print('test MAPE = ', score_test[3])
        result_test = keypoint_model.predict(featuremap_test)

        # Plot accuracy
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        layer_type = 'convolutional' if validation_type == 'convolutional' else 'dense'
        plt.title('Model accuracy for ' + str(n) + ' ' + layer_type + ' layer(s)')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Xval'], loc='upper left')
        plt.savefig('model/plots/accuracy_' + str(n) + '.png')
        plt.show()
        plt.close()

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss for ' + str(n) + ' ' + layer_type + ' layer(s)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Xval'], loc='upper left')
        plt.xlim([0,100])
        plt.ylim([0,0.1])
        plt.savefig('model/plots/loss_' + str(n) + '.png')
        plt.show()
        plt.close()

        # error for each axis
        print("mae for x is",metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19]))
        print("mae for y is",metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38]))
        print("mae for z is",metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57]))

        # matrix transformation for the final all 19 points mae
        x_mae = metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values')
        y_mae = metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values')
        z_mae = metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values')

        all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3,19)
        avg_19_points_mae = np.mean(all_19_points_mae, axis = 0)
        avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis = 1).reshape(1,3)

        all_19_points_mae_Transpose = all_19_points_mae.T

        # matrix transformation for the final all 19 points rmse
        x_rmse = metrics.mean_squared_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values', squared=False)
        y_rmse = metrics.mean_squared_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values', squared=False)
        z_rmse = metrics.mean_squared_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values', squared=False)

        all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3,19)
        avg_19_points_rmse = np.mean(all_19_points_rmse, axis = 0)
        avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis = 1).reshape(1,3)

        all_19_points_rmse_Transpose = all_19_points_rmse.T

        # merge the mae and rmse
        all_19_points_maermse_Transpose = np.concatenate((all_19_points_mae_Transpose,all_19_points_rmse_Transpose), axis = 1)*100
        avg_19_points_maermse_Transpose = np.concatenate((avg_19_points_mae_xyz,avg_19_points_rmse_xyz), axis = 1)*100

        # concatenate the array, the final format is the same as shown in paper.
        # First 19 rows each joint, the final row is the average
        paper_result_maermse = np.concatenate((all_19_points_maermse_Transpose, avg_19_points_maermse_Transpose), axis = 0)
        paper_result_maermse = np.around(paper_result_maermse, 2)
        # reorder the columns to make it xmae, xrmse, ymae, yrmse, zmae, zrmse, avgmae, avgrmse
        paper_result_maermse = paper_result_maermse[:, [0,3,1,4,2,5]]

        # append each iteration's result
        paper_result_list.append(paper_result_maermse)

        # define the output directory
        output_direct = 'model/'

        if not os.path.exists(output_direct):
            os.makedirs(output_direct)

        # save the best model so far
        if score_test[1] < score_min:
            keypoint_model.save(output_direct + 'MARS_' + str(n) + '.h5')
            score_min = score_test[1]

        avg_mae.append(history.history['mae'][-1])
        avg_val_mae.append(history.history['val_mae'][-1])
        avg_loss.append(history.history['loss'][-1])
        avg_val_loss.append(history.history['val_loss'][-1])

    times.append(np.mean(conv_times))

    # average the result for all iterations
    mean_paper_result_list = np.mean(paper_result_list, axis = 0)
    mean_mae = np.mean(np.dstack((mean_paper_result_list[:,0], mean_paper_result_list[:,2], mean_paper_result_list[:,4])).reshape(20,3), axis = 1)
    mean_rmse = np.mean(np.dstack((mean_paper_result_list[:,1], mean_paper_result_list[:,3], mean_paper_result_list[:,5])).reshape(20,3), axis = 1)
    mean_paper_result_list = np.concatenate((np.mean(paper_result_list, axis = 0), mean_mae.reshape(20,1), mean_rmse.reshape(20,1)), axis = 1)

    # Add the average mae and loss to the list
    avg_mae_list.append(np.mean(avg_mae))
    print(f"Average Training MAE for {n} conv layers: {np.mean(avg_mae)}")
    avg_val_mae_list.append(np.mean(avg_val_mae))
    avg_loss_list.append(np.mean(avg_loss))
    avg_val_loss_list.append(np.mean(avg_val_loss))

    # Export the Accuracy
    output_path = output_direct + "Accuracy"
    output_filename = output_path + "/MARS_accuracy_" + str(n)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(output_filename + ".npy", mean_paper_result_list)
    np.savetxt(output_filename + ".txt", mean_paper_result_list,fmt='%.2f')

# Plot the average MAEs, Losses and Time Taken across different numbers of convolutional layers
if validation_type == 'convolutional':
    # Plot the average MAEs across different numbers of convolutional layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_conv_layers, avg_val_mae_list, marker='o', linestyle='-', color='b')
    plt.title('Average Model Validation MAE by Number of Convolutional Layers')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Average MAE')
    plt.xticks(num_conv_layers)
    plt.grid(True)
    plt.savefig('model/plots/avg_mae.png')
    plt.show()
    plt.close()

    # Plot the average Losses across different numbers of convolutional layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_conv_layers, avg_val_loss_list, marker='o', linestyle='-', color='r')
    plt.title('Average Model Validation Loss by Number of Convolutional Layers')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Average Loss')
    plt.xticks(num_conv_layers)
    plt.grid(True)
    plt.savefig('model/plots/avg_loss.png')
    plt.show()
    plt.close()

    # Plot the time taken to train the model across different numbers of convolutional layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_conv_layers, times, marker='o', linestyle='-', color='g')
    plt.title('Time Taken to Train Model by Number of Convolutional Layers')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Time Taken (s)')
    plt.xticks(num_conv_layers)
    plt.grid(True)
    plt.savefig('model/plots/time_taken.png')
    plt.show()
    plt.close()

# Plot the average MAEs, Losses and Time Taken across different numbers of dense layers
elif validation_type == 'dense':
    # Plot the average MAEs across different numbers of dense layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_dense_layers, avg_val_mae_list, marker='o', linestyle='-', color='b')
    plt.title('Average Model Validation MAE by Number of Dense Layers')
    plt.xlabel('Number of Dense Layers')
    plt.ylabel('Average MAE')
    plt.xticks(num_dense_layers)
    plt.grid(True)
    plt.savefig('model/plots/avg_mae.png')
    plt.show()
    plt.close()

    # Plot the average Losses across different numbers of dense layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_dense_layers, avg_val_loss_list, marker='o', linestyle='-', color='r')
    plt.title('Average Model Validation Loss by Number of Dense Layers')
    plt.xlabel('Number of Dense Layers')
    plt.ylabel('Average Loss')
    plt.xticks(num_dense_layers)
    plt.grid(True)
    plt.savefig('model/plots/avg_loss.png')
    plt.show()
    plt.close()

    # Plot the time taken to train the model across different numbers of dense layers
    plt.figure(figsize=(10, 5))
    plt.plot(num_dense_layers, times, marker='o', linestyle='-', color='g')
    plt.title('Time Taken to Train Model by Number of Dense Layers')
    plt.xlabel('Number of Dense Layers')
    plt.ylabel('Time Taken (s)')
    plt.xticks(num_dense_layers)
    plt.grid(True)
    plt.savefig('model/plots/time_taken.png')
    plt.show()
    plt.close()
