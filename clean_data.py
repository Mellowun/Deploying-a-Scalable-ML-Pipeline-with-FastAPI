import pandas as pd

def clean_csv_spaces(input_file, output_file):
    """
    Reads a CSV file, removes leading/trailing spaces from all string columns,
    and saves the cleaned data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_file)

        # Iterate through columns and strip spaces from string columns
        for col in df.columns:
            if df[col].dtype == 'object':  # Check if the column is a string
                df[col] = df[col].str.strip() #remove leading and trailing spaces
                # df[col] = df[col].str.replace(' ', '') #remove all spaces

        df.to_csv(output_file, index=False)
        print(f"Cleaned CSV saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_csv = "C:/Users/mello/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv"  # Replace with your input file name
output_csv = "cleaned_output.csv" # Replace with your output file name

clean_csv_spaces(input_csv, output_csv)