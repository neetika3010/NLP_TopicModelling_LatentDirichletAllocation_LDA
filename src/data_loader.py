import pandas as pd

def load_data(file_path, text_column='text'):
    """
    Load the dataset from CSV or Excel and return the text column.

    Args:
        file_path (str): Path to the file.
        text_column (str): Name of the text column.

    Returns:
        List[str]: List of raw text documents.
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in file")

    return df[text_column].dropna().tolist()
