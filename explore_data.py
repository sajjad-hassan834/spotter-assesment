import pandas as pd

def explore_data(filepath):
    print(f"\n--- Exploring {filepath} ---")
    df = pd.read_csv(filepath)
    print("Shape:", df.shape)
    print("\nColumns and Data Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe(include='all'))

if __name__ == '__main__':
    explore_data('data/train_test.csv')
    explore_data('data/validation.csv')
