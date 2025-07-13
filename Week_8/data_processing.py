
import pandas as pd

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    # For now, let's convert each row into a descriptive string.
    # This can be refined later based on specific Q&A needs.
    df_text = df.apply(lambda x: ', '.join([f'{col}: {val}' for col, val in x.items()]), axis=1)
    return df_text.tolist()

if __name__ == '__main__':
    training_data = load_and_process_data('/home/ubuntu/upload/TrainingDataset.csv')
    test_data = load_and_process_data('/home/ubuntu/upload/TestDataset.csv')
    
    print(f"Processed Training Data Samples (first 2):\n{training_data[:2]}")
    print(f"Processed Test Data Samples (first 2):\n{test_data[:2]}")


