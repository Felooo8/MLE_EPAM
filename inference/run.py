"""
Script loads the latest trained model, data for inference, and predicts results.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Configuration file path
CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, "settings.json"))

# Load configuration settings
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = os.path.abspath(conf['general']['data_dir'])
MODEL_DIR = os.path.abspath(conf['general']['models_dir'])
RESULTS_DIR = os.path.abspath(conf['general']['results_dir'])

# Initialize parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model."""
    latest = None
    for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".pickle"):
                current_time = datetime.strptime(filename.split('.')[0], conf['general']['datetime_format'])
                if not latest or datetime.strptime(latest, conf['general']['datetime_format']) < current_time:
                    latest = filename.split('.')[0]
    if latest is None:
        raise FileNotFoundError("No model file found in the model directory.")
    return os.path.join(MODEL_DIR, f"{latest}.pickle")


def load_model(path: str) -> RandomForestClassifier:
    """Loads and returns the trained model."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
            logging.info(f"Loaded model from: {path}")
            return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit(1)


def load_inference_data(path: str) -> pd.DataFrame:
    """Loads and returns data for inference from the specified CSV file."""
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f"Error loading inference data: {e}")
        sys.exit(1)


def predict_results(model: RandomForestClassifier, data: pd.DataFrame) -> pd.DataFrame:
    """Predicts results using the model and appends predictions to the dataset."""
    try:
        features = data.drop(columns=['target'], errors='ignore')  # Drop 'target' if it exists
        data['predictions'] = model.predict(features)
        return data
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        sys.exit(1)


def save_results(results: pd.DataFrame, path: str = None) -> None:
    """Saves prediction results to a CSV file."""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = os.path.join(RESULTS_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.csv')
    try:
        results.to_csv(path, index=False)
        logging.info(f"Results saved to: {path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        sys.exit(1)


def main():
    """Main function to execute inference."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting inference script...")

    args = parser.parse_args()

    try:
        model_path = get_latest_model_path()
        model = load_model(model_path)
        
        infer_file = os.path.join(DATA_DIR, args.infer_file)
        data = load_inference_data(infer_file)
        
        results = predict_results(model, data)
        save_results(results, args.out_path)
        
        logging.info("Inference completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
