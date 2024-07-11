import os
import re
import pandas as pd

def extract_accuracies(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Training accuracy of model after training for 1 epochs" in line:
                train_accuracy = float(re.search(r"Training accuracy of model after training for 1 epochs: ([0-9.]+)", line).group(1))
            if "Testing accuracy of model after training for 1 epochs" in line:
                test_accuracy = float(re.search(r"Testing accuracy of model after training for 1 epochs: ([0-9.]+)", line).group(1))
    return train_accuracy, test_accuracy

def extract_learning_rate(folder_name):
    match = re.search(r'(\d+\.\d+)', folder_name)
    return float(match.group(1)) if match else None

def process_experiment_folders(base_path):
    data = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            log_file_path = os.path.join(folder_path, 'parameters.log')
            if os.path.exists(log_file_path):
                train_acc, test_acc = extract_accuracies(log_file_path)
                learning_rate = extract_learning_rate(folder_name)
                if learning_rate is not None:
                    data.append((folder_name, learning_rate, train_acc, test_acc))
    
    # Sort data by learning rate
    data.sort(key=lambda x: x[1])
    return data

def save_to_excel(data, output_path):
    df = pd.DataFrame(data, columns=['Experiment', 'Learning Rate', 'Train Accuracy', 'Test Accuracy'])
    df.to_excel(output_path, index=False)

# Define the base path containing experiment folders
base_path = '../Table-Related'  # Change this to your base directory path
output_path = 'experiment_accuracies.xlsx'

# Process the folders and save the results to an Excel file
experiment_data = process_experiment_folders(base_path)
save_to_excel(experiment_data, output_path)

print(f"Excel file '{output_path}' created successfully.")