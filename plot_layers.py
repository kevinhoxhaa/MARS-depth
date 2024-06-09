import matplotlib.pyplot as plt
import numpy as np


class PlotLayers:
    def __init__(self, num_conv_layers, num_dense_layers, avg_val_mae_list, avg_val_loss_list, boxplot_mae_list, boxplot_loss_list, times):
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.avg_val_mae_list = avg_val_mae_list
        self.avg_val_loss_list = avg_val_loss_list
        self.boxplot_mae_list = boxplot_mae_list
        self.boxplot_loss_list = boxplot_loss_list
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

        # Plot the boxplots of the MAEs across different numbers of convolutional layers
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(self.boxplot_mae_list, patch_artist=True)
        plt.title('Model Validation MAE by Number of Convolutional Layers')
        plt.xlabel('Number of Convolutional Layers')
        plt.ylabel('MAE')
        plt.xticks(self.num_conv_layers)
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.savefig('model/plots/boxplot_mae.png')
        plt.show()
        plt.close()

        # Plot the boxplots of the Losses across different numbers of convolutional layers
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(self.boxplot_loss_list, patch_artist=True)
        plt.title('Model Validation Loss by Number of Convolutional Layers')
        plt.xlabel('Number of Convolutional Layers')
        plt.ylabel('Loss')
        plt.xticks(self.num_conv_layers)
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.savefig('model/plots/boxplot_loss.png')
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

        # Plot the boxplots of the MAEs across different numbers of dense layers
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(self.boxplot_mae_list, patch_artist=True)
        plt.title('Model Validation MAE by Number of Dense Layers')
        plt.xlabel('Number of Dense Layers')
        plt.ylabel('MAE')
        plt.xticks(self.num_dense_layers)
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.savefig('model/plots/boxplot_mae.png')
        plt.show()
        plt.close()

        # Plot the boxplots of the Losses across different numbers of dense layers
        plt.figure(figsize=(10, 5))
        bplot = plt.boxplot(self.boxplot_loss_list, patch_artist=True)
        plt.title('Model Validation Loss by Number of Dense Layers')
        plt.xlabel('Number of Dense Layers')
        plt.ylabel('Loss')
        plt.xticks(self.num_dense_layers)
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.savefig('model/plots/boxplot_loss.png')
        plt.show()
        plt.close()

def plot_boxplots():
    '''
    Plot the boxplots of the MAEs and Losses across different numbers of convolutional and dense layers
    '''
    # Load the data
    mae_boxplots = [
        100 * np.load('model/Accuracy/normal_MARS/MARS_accuracy_2_mae.npy')[0],
        100 * np.load('model/Accuracy/optimal_MARS/MARS_accuracy_2_mae.npy')[0],
        100 * np.load('model/Accuracy/pointnet/PointNet_accuracy_mae.npy')[0],
    ]

    rmse_boxplots = [
        100 * np.load('model/Accuracy/normal_MARS/MARS_accuracy_2_rmse.npy')[0],
        100 * np.load('model/Accuracy/optimal_MARS/MARS_accuracy_2_rmse.npy')[0],
        100 * np.load('model/Accuracy/pointnet/PointNet_accuracy_rmse.npy')[0],
    ]

    # Colors as seen in your example
    colors = ['lightblue', 'lightgreen', 'lightpink']

    model_names = ['Defualt MARS', 'Optimal MARS', 'PointNet']

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Positions for the groups
    positions_A = np.array([1, 1.75, 2.5])
    positions_B = np.array([3.5, 4.25, 5])

    # Create boxplots for Group A
    for pos, data, color in zip(positions_A, mae_boxplots, colors):
        bplot = ax.boxplot(data, positions=[pos], widths=0.5, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # Create boxplots for Group B
    for pos, data, color in zip(positions_B, rmse_boxplots, colors):
        bplot = ax.boxplot(data, positions=[pos], widths=0.5, patch_artist=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # Adding custom x-tick labels
    tick_positions = [np.mean(positions_A), np.mean(positions_B)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(['MAE', 'RMSE'])

    # Adding a legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'{model_names[i]}') for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Draw a vertical line between the two groups to separate them visually
    ax.axvline(x=3, color='grey', linestyle='--', linewidth=2)  # Adjust 'x' as necessary to align with your plot

    # Final plot adjustments
    ax.set_title('Error Comparison Between Models')
    ax.set_ylabel('Error (cm)')
    ax.set_xlabel('Error Metric')
    # ax.grid(axis='y')
    # Show plot
    plt.show()
    plt.close()

plot_boxplots()
