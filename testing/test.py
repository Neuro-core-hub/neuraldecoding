import pandas as pd

def compare_csvs(file1, file2):
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # Check if both have the same columns (ignoring order)
        if set(df1.columns) != set(df2.columns):
            return False

        # Reorder columns to match
        df2 = df2[df1.columns]

        # Sort rows to ignore row order differences
        df1 = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
        df2 = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)

        return df1.equals(df2)
    
    except Exception as e:
        print(f"Error: {e}")
        return False

# Example usage
file1 = "output_data.csv"
file2 = "output_data_optimized.csv"
print(compare_csvs(file1, file2))
