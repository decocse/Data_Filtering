Financial Data Validation System

Overview

This project is a Regulatory Reporting Validation System that uses a combination of rule-based validation and AI-driven anomaly detection to ensure financial dataset integrity. It leverages a lightweight open-source language model (FLAN-T5) to generate validation rules and Isolation Forest to detect anomalies.

Features

Automated Validation Rule Generation: Uses an LLM to generate and explain financial validation rules.

Rule-Based Data Validation: Checks for missing values, invalid entries, and negative values where applicable.

AI-Powered Anomaly Detection: Detects outliers in financial data using Isolation Forest.

Detailed Reporting: Generates a Word report summarizing the errors, anomalies, and validation rules.

Prerequisites

Before running the script, install the required dependency:

pip install python-docx

Running the Code in Google Colab

Upload your financial dataset (CSV or Excel file) to Google Colab.

Run the script and provide the file path when prompted.

📂 Enter the path to the CSV/Excel file: /content/myfile.xlsx

Expected Output

A validation report (validation_report.docx) with:

Generated validation rules and explanations.

Errors found in the dataset (missing values, incorrect data types, etc.).

Anomaly descriptions with specific details on suspicious values.

Example Anomaly Output

🚨 Row 12 is an anomaly. Suspicious values: Total_Assets: 50,000,000, Net_Profit: -5,000,000
🚨 Row 27 is an anomaly. Suspicious values: Liabilities: 200,000,000

Notes

Works best with structured financial data.

Can be extended with additional rule sets for compliance.

Next Steps

Improve rule generation with larger models.

Enhance anomaly detection using ensemble methods.

Expand validation rules for industry-specific compliance.
