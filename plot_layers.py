import matplotlib.pyplot as plt


class PlotLayers:
    def __init__(self, num_conv_layers, num_dense_layers, avg_val_mae_list, avg_val_loss_list, times):
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.avg_val_mae_list = avg_val_mae_list
        self.avg_val_loss_list = avg_val_loss_list
        self.times = times

    def plot_conv(self):
        '''
        Plot the average MAEs, Losses and Time Taken across different numbers of convolutional layers
        '''
        # Plot the average MAEs across different numbers of convolutional layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_conv_layers, self.avg_val_mae_list, marker='o', linestyle='-', color='b')
        plt.title('Average Model Validation MAE by Number of Convolutional Layers')
        plt.xlabel('Number of Convolutional Layers')
        plt.ylabel('Average MAE')
        plt.xticks(self.num_conv_layers)
        plt.grid(True)
        plt.savefig('model/plots/avg_mae.png')
        plt.show()
        plt.close()

        # Plot the average Losses across different numbers of convolutional layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_conv_layers, self.avg_val_loss_list, marker='o', linestyle='-', color='r')
        plt.title('Average Model Validation Loss by Number of Convolutional Layers')
        plt.xlabel('Number of Convolutional Layers')
        plt.ylabel('Average Loss')
        plt.xticks(self.num_conv_layers)
        plt.grid(True)
        plt.savefig('model/plots/avg_loss.png')
        plt.show()
        plt.close()

        # Plot the time taken to train the model across different numbers of convolutional layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_conv_layers, self.times, marker='o', linestyle='-', color='g')
        plt.title('Time Taken to Train Model by Number of Convolutional Layers')
        plt.xlabel('Number of Convolutional Layers')
        plt.ylabel('Time Taken (s)')
        plt.xticks(self.num_conv_layers)
        plt.grid(True)
        plt.savefig('model/plots/time_taken.png')
        plt.show()
        plt.close()

    def plot_dense(self):
        '''
        Plot the average MAEs, Losses and Time Taken across different numbers of dense layers
        '''
        # Plot the average MAEs across different numbers of dense layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_dense_layers, self.avg_val_mae_list, marker='o', linestyle='-', color='b')
        plt.title('Average Model Validation MAE by Number of Dense Layers')
        plt.xlabel('Number of Dense Layers')
        plt.ylabel('Average MAE')
        plt.xticks(self.num_dense_layers)
        plt.grid(True)
        plt.savefig('model/plots/avg_mae.png')
        plt.show()
        plt.close()

        # Plot the average Losses across different numbers of dense layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_dense_layers, self.avg_val_loss_list, marker='o', linestyle='-', color='r')
        plt.title('Average Model Validation Loss by Number of Dense Layers')
        plt.xlabel('Number of Dense Layers')
        plt.ylabel('Average Loss')
        plt.xticks(self.num_dense_layers)
        plt.grid(True)
        plt.savefig('model/plots/avg_loss.png')
        plt.show()
        plt.close()

        # Plot the time taken to train the model across different numbers of dense layers
        plt.figure(figsize=(10, 5))
        plt.plot(self.num_dense_layers, self.times, marker='o', linestyle='-', color='g')
        plt.title('Time Taken to Train Model by Number of Dense Layers')
        plt.xlabel('Number of Dense Layers')
        plt.ylabel('Time Taken (s)')
        plt.xticks(self.num_dense_layers)
        plt.grid(True)
        plt.savefig('model/plots/time_taken.png')
        plt.show()
        plt.close()
