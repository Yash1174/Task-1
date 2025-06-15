import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# ------------------------------
# Configuration
# ------------------------------
INPUT_CSV = 'data/raw_data.csv'
OUTPUT_CSV = 'data/processed_data.csv'

# Define columns
NUMERIC_FEATURES = ['age', 'income']
CATEGORICAL_FEATURES = ['gender', 'occupation']

# ------------------------------
# Step 1: Extract
# ------------------------------
def extract_data(filepath):
    print("Extracting data...")
    return pd.read_csv(filepath)

# ------------------------------
# Step 2: Transform
# ------------------------------
def build_transform_pipeline():
    print("Building transformation pipeline...")

    # Pipelines for different column types
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor

def transform_data(df, pipeline):
    print("Transforming data...")
    transformed_array = pipeline.fit_transform(df)
    feature_names = pipeline.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed_array.toarray() if hasattr(transformed_array, "toarray") else transformed_array, 
                                  columns=feature_names)
    return transformed_df

# ------------------------------
# Step 3: Load
# ------------------------------
def load_data(df, output_path):
    print(f"Loading data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Load complete.")

# ------------------------------
# Main ETL Function
# ------------------------------
def run_etl(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    df = extract_data(input_path)
    pipeline = build_transform_pipeline()
    transformed_df = transform_data(df, pipeline)
    load_data(transformed_df, output_path)

# ------------------------------
# Entry Point
# ------------------------------
if __name__ == '__main__':
    run_etl()
