import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_silo(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Drop patient ID or any leakage columns
    df.drop(columns=['encounter_id', 'patient_nbr'], inplace=True, errors='ignore')

    # Drop rows with missing target if any
    df = df.dropna(subset=['readmitted'])

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(['readmitted'])
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed: {os.path.basename(input_path)} → {os.path.basename(output_path)} | Rows: {len(df)}")

# Apply to all silos
input_dir = '../datasets/diabetes/silos'
output_dir = '../datasets/diabetes/processed_silos'

for file in os.listdir(input_dir):
    if file.endswith('.csv'):
        preprocess_silo(
            input_path=os.path.join(input_dir, file),
            output_path=os.path.join(output_dir, file)
        )
