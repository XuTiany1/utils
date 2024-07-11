import re
import pandas as pd

def extract_results(log_file_path, output_excel_path):
    # Regular expression to match the required lines and extract values
    pattern = re.compile(r"Lambda: ([\d\.]+) \|\|.*?Test Acc: avg = ([\d\.]+), var = ([\d\.e\-]+) \|\| Train Acc: avg = ([\d\.]+), var = ([\d\.e\-]+)")
    
    data = {
        'Lambda': [],
        'Avg Train Accuracy': [],
        'Train Variance': [],
        'Avg Test Accuracy': [],
        'Test Variance': []
    }

    # Read the log file and extract values
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                lambda_value = float(match.group(1))
                avg_test_acc = float(match.group(2)) * 100
                test_variance = float(match.group(3))
                avg_train_acc = float(match.group(4)) * 100
                train_variance = float(match.group(5))
                
                data['Lambda'].append(lambda_value)
                data['Avg Train Accuracy'].append(avg_train_acc)
                data['Train Variance'].append(train_variance)
                data['Avg Test Accuracy'].append(avg_test_acc)
                data['Test Variance'].append(test_variance)
            else:
                print(f"No match found in line: {line}")

    # Check if data is being collected properly
    print(data)

    # Create a DataFrame with the extracted values
    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)

# Example usage
log_file_path = '/mnt/data/results.log'  # Path to the uploaded log file
output_excel_path = '/mnt/data/results_summary.xlsx'
extract_results(log_file_path, output_excel_path)

