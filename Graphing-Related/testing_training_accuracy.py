import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re
import os


def parse_log(file_path, accuracy_type):
    """
    This basically reads the log file and puts the samples and the accuracies in order in two arrays
    This returns two arrays:
        the first array will store the samples seen
        the second array will store the respective accuracies
    """
    
    # Debugging: Verify file path
    print(f"Parsing log file: {file_path}")

    # First, I open the filepath in read mode and read all the liens
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Debugging: Verify content read
    print(f"Read {len(data)} lines from {file_path}")

    # Now, I initiate two empty arrays
    samples = []
    accuracies = []

    # Now, I loop through each line of the read liens
    for line in data:

        # Now, I use regex expression to find a match
        match = re.search(r'Samples Seen: (\d+) .* {}: ([0-9.]+)'.format(accuracy_type), line)
            # the "re.search" uses a regular expression to search for patterns in each line that match the given format
            # r'Samples Seen: (\d+) .* {}: ([0-9.]+)': This pattern is used to extract the number of samples seen and the accuracy value
                # (\d+): Matches one or more digits (representing the number of samples seen).
                # .*: Matches any character (except for a newline) zero or more times.
                # {}: ([0-9.]+): Inserts the accuracy_type into the pattern and matches one or more digits or a period (representing the accuracy value).

        if match:
            samples.append(int(match.group(1)))             # match.group(1): Retrieves the first matched group (the number of samples seen).
            accuracies.append(float(match.group(2)))        # match.group(2): Retrieves the second matched group (the accuracy value).

    return samples, accuracies




def create_excel(writer, sheet_name, samples, training_accuracy, testing_accuracy, train_eq, test_eq):    
    """
    This function converts the samples, training, and testing accuracies into a table
    Then, it outputs into excel 

    @param
        samples: a list of integers representing the number of samples seen
        training_accuracy: a list of float representing training accuracy values corresponding to each number of sampels seen
        testing_accuracy: a list of float representing testing accuracy values corresponding to each number of sampels seen
        output_path: a string representing the path where excel file should be saved
    @return
        None
    """

    # First, I create a dataframe object from a dictionary
        # The keys are the column names
        # The values are the lists provided as function arguments
    df = pd.DataFrame({
        'Number of Samples Seen': samples,
        'Training Accuracy': training_accuracy,
        'Testing Accuracy': testing_accuracy
    })
    # Here, the resulting datafram will have three columns:
    # COLUMN 1 -> Number of Samples Seen
    # COLUMN 2 -> Training Accuracy
    # COLUMN 3 -> TESTING ACCURACY

    # Create a new DataFrame for the equations
    eq_df = pd.DataFrame({
        'Type': ['Training', 'Testing'],
        'Equation': [train_eq, test_eq]
    })


    # Write both DataFrames to the Excel file
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
    eq_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(df) + 2)

    print(f"Data for {sheet_name} written to Excel.")


def log_fit_curve(x,a,b):
    """
    Here, I define a function for the best-fit curve for log
    This function basically models the data as an log curve plus a constant
    """
    return a * np.log(x + 1) + b        # 1 is added to avoid division by 0


def plot_accuracies(samples, training_accuracy, testing_accuracy, experiment_name, output_folder):
    """
    Here, I create the plot that will plot samples as x-axis and the training + testing accuracy as y-axis
    """

    # First, I setup the canvas for graphing as 10 inches by 6 inches
    plt.figure(figsize=(10,6))

    # Now, I plot the lines
        # samples = x-axis
        # train/test accuracy = y-axis
        # b-/r- means blue/red values
        # label is straight forward
    plt.plot(samples, training_accuracy, 'b-', label='Training Accuracy') 
    plt.plot(samples, testing_accuracy, 'r-', label='Testing Accuracy')


    # Now, I curve fit using the previously defined curve defiend in log_fit_curve function
    # Here, the optimal_param will contain the best-fit values of a and b for the training and testing data
    # and, the _ will be the estimated covariance of the optimal parameters -> this is not really used so I assign it to _
    optimal_param_train, _ = curve_fit(log_fit_curve, samples, training_accuracy, maxfev=10000)
    optimal_param_test, _ = curve_fit(log_fit_curve, samples, testing_accuracy, maxfev=10000)


    # Lastly, I use the optimal parameters to plot these best-fit curves
    plt.plot(samples, log_fit_curve(np.array(samples), *optimal_param_train), 'b--', label='Best Log Fit Train')
    plt.plot(samples, log_fit_curve(np.array(samples), *optimal_param_test), 'r--', label='Best Log Fit Test')

    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Number of Samples Seen (log scale)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training and Testing Accuracy for {experiment_name}')
    plt.grid(True)
    # Add equations to the plot
    train_eq = f'Training: y = {optimal_param_train[0]:.4f} * log(x + 1) + {optimal_param_train[1]:.4f}'
    test_eq = f'Testing: y = {optimal_param_test[0]:.4f} * log(x + 1) + {optimal_param_test[1]:.4f}'
    plt.text(0.05, 0.95, train_eq, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='blue', facecolor='white'))
    plt.text(0.05, 0.85, test_eq, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='red', facecolor='white'))


    # Save the plot
    plot_filename = os.path.join(output_folder, f'{experiment_name}_accuracy_plot.png')
    plt.savefig(plot_filename)
    plt.close()

    return train_eq, test_eq, plot_filename


def process_experiment_folder(experiment_folder, writer, output_folder):
    """
    This function

    @param
        experiment_folder: The path to the current experiment folder being processed.
        writer: The Excel writer object used to write data to different sheets in the Excel file.
        output_folder: The path to the folder where the plots will be saved.
    @return
        None
    """

    # First, I construct path to training and testing accuarcy log files
    training_log = os.path.join(experiment_folder, 'training_accuracy.log')
    testing_log = os.path.join(experiment_folder, 'testing_accuracy.log')

    # Now, I extract the name of the experiment folder
    experiment_name = os.path.basename(experiment_folder)

    # Debugging prints
    print(f"Processing folder: {experiment_folder}")
    print(f"Training log path: {training_log}")
    print(f"Testing log path: {testing_log}")

    # DEBUGGING PURPOSES
    if not os.path.isfile(training_log) or not os.path.isfile(testing_log):
        print(f"Log files not found for {experiment_name}. Skipping...")
        return
    
    # Now, I get the samples train and training/testing accuracies
    samples_train, training_accuracy = parse_log(training_log, 'Train Accuracy')
    samples_test, testing_accuracy = parse_log(testing_log, 'Test Accuracy')

    # Ensure both logs have the same samples seen
    samples = sorted(set(samples_train) & set(samples_test))
    # Filters the training and testing accuracy values to include only those corresponding to the matching sample points.
    training_accuracy = [training_accuracy[samples_train.index(s)] for s in samples]
    testing_accuracy = [testing_accuracy[samples_test.index(s)] for s in samples]
    
    # I plot and I get the best fit equations as well as the plot file name (this file name will already have a plot saved inside)
    train_eq, test_eq, plot_filename = plot_accuracies(samples, training_accuracy, testing_accuracy, experiment_name, output_folder)

    create_excel(writer, experiment_name, samples, training_accuracy, testing_accuracy, train_eq, test_eq)


def get_experiment_folders(base_folder):
    """
    Get a list of paths to all experiment folders in the base folder
    @param
        base_folder: the path to the base directory containing the experiment folders
    """
    # List all items in the base folder
    items = os.listdir(base_folder)
    # Filter out items that are directories and return their full paths
    experiment_folders = [
        os.path.join(base_folder, item)                     # os.listdir(base_folder) returns a list of all items (files and directories) in base_folder.
        for item in items                                   # for item in items iterates over each item in the items list obtained from os.listdir.
        if os.path.isdir(os.path.join(base_folder, item))   # os.path.isdir(os.path.join(base_folder, item)) checks if the current item is a directory.
    ]
    """
    The list comprehension iterates over each item in items.
	    For each item, it checks if the item is a directory using os.path.isdir.
        If the item is a directory, it constructs the full path using os.path.join(base_folder, item).
        The resulting list, experiment_folders, contains the full paths of all directories within base_folder.
    """


    return experiment_folders

def main(base_folder, output_excel):
    """
    @param
        base_folder: the path to the base directory containing the experiment folders
        output_excel: the path to the output excel file where results will be saved
    @return 
        NONE
    
    """
    output_folder = 'result'
    # Creates the ‘result’ directory if it does not exist, using os.makedirs with exist_ok=True.
    os.makedirs(output_folder, exist_ok=True)

    # Debugging: Verify base_folder path
    print(f"Base folder: {base_folder}")

    experiment_folders = get_experiment_folders(base_folder)

    # Debugging: Check experiment folders
    print("Experiment folders found:")
    for folder in experiment_folders:
        print(folder)

    with pd.ExcelWriter(output_excel) as writer:
        for experiment_folder in experiment_folders:
            # Debugging: Track processing of each experiment folder
            print(f"Processing experiment folder: {experiment_folder}")
            process_experiment_folder(experiment_folder, writer, output_folder)

    print(f"All data written to {output_excel}. Plots saved in {output_folder}.")

# Example usage
base_folder = '../ortho-relu-lin'
output_excel = '../accuracy_comparison.xlsx'  # Save the Excel file one directory up from the script

main(base_folder, output_excel)









































































