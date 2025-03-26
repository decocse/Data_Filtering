import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.ensemble import IsolationForest

# Load a small open-source LLM
MODEL_NAME = "google/flan-t5-small"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

def load_data(file_path):
    """Load the dataset from CSV or Excel."""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def generate_rules():
    """Use LLM to generate validation rules."""
    prompt = "Generate validation rules for a financial dataset with columns: Company_ID, Total_Assets, Liabilities, Net_Profit, Regulatory_Compliance."

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")  # Use CPU

    # Generate text
    with torch.no_grad():
        output = model.generate(inputs, max_length=200)

    # Decode response
    rules = tokenizer.decode(output[0], skip_special_tokens=True)
    return rules

def validate_data(df):
    """Perform rule-based validation and detect anomalies."""
    errors = []

    # Check for missing values
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            errors.append(f"Column '{column}' has {missing_count} missing values.")

    # Check for negative financial values
    for column in ["Total_Assets", "Liabilities", "Net_Profit"]:
        if column in df.columns and (df[column] < 0).any():
            errors.append(f"Column '{column}' contains negative values.")

    # Check compliance status values
    if "Regulatory_Compliance" in df.columns:
        valid_status = ["Compliant", "Non-Compliant"]
        if not df["Regulatory_Compliance"].isin(valid_status).all():
            errors.append("Invalid values detected in 'Regulatory_Compliance' column.")

    return errors

def detect_anomalies(df):
    """Use Isolation Forest to detect anomalies in financial data."""
    anomalies = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return ["No numeric columns found for anomaly detection."]

    model = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(df[numeric_cols])

    anomaly_rows = df[df["Anomaly"] == -1]
    if not anomaly_rows.empty:
        anomalies.append(f"Detected {len(anomaly_rows)} anomalies in the dataset.")

    return anomalies

def generate_report(errors, anomalies, rules, report_path="validation_report.txt"):
    """Generate and save a detailed validation report."""
    with open(report_path, "w") as report_file:
        report_file.write("Generated Validation Rules:\n")
        report_file.write(rules + "\n\n")

        if errors:
            report_file.write("Basic Validation Errors:\n")
            report_file.write("\n".join(errors))
            report_file.write("\n\n")

        if anomalies:
            report_file.write("AI-Driven Anomaly Detection:\n")
            report_file.write("\n".join(anomalies))
        else:
            report_file.write("No anomalies detected.")

    print(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    file_path = input("Enter the path to the CSV/Excel file: ")
    df = load_data(file_path)

    if df is not None:
        rules = generate_rules()
        validation_errors = validate_data(df)
        anomaly_results = detect_anomalies(df)
        generate_report(validation_errors, anomaly_results, rules)
