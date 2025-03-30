# filepath: ga_dissertation/data_processor.py
import pandas as pd
import numpy as np
import os


class DataProcessor:
    def __init__(self, case_study_dir):
        self.case_study_dir = case_study_dir

    def load_case_study_data(self, filename):
        """Load data from a case study file"""
        file_path = os.path.join(self.case_study_dir, filename)

        # Determine file type and load accordingly
        if filename.endswith(".csv"):
            return pd.read_csv(file_path)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(file_path)
        elif filename.endswith(".txt"):
            # For simple text files, try to infer structure
            return pd.read_csv(file_path, delimiter="\t")
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def extract_parameters(self, data, parameter_names):
        """Extract specific parameters from the data"""
        parameters = {}

        for param in parameter_names:
            if param in data.columns:
                parameters[param] = data[param].values
            else:
                print(f"Warning: Parameter {param} not found in data")

        return parameters

    def preprocess_data(self, data):
        """Preprocess data for GA (e.g., normalization, handling missing values)"""
        # Handle missing values
        data = data.fillna(data.mean())

        # Normalize numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = (data[col] - data[col].min()) / (
                data[col].max() - data[col].min()
            )

        return data
