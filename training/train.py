"""
This script prepares the Iris dataset, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Optional: Uncomment this line if using MLFlow for tracking
# import mlflow
# mlflow.autolog()

# Add root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Configuration file path (use environment variable or default to settings.json)
CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, "settings.json"))

# Load configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = os.path.abspath(conf['general']['data_dir'])
MODEL_DIR = os.path.abspath(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initialize parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="Specify training data file", default=conf['train']['table_name'])
parser.add_argument("--model_path", help="Specify the path for the output model")


class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info("Max_rows not defined. Skipping sampling.")
        elif len(df) < max_rows:
            logging.info("Size of dataframe is less than max_rows. Skipping sampling.")
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f"Random sampling performed. Sample size: {max_rows}")
        return df


class Training:
    def __init__(self) -> None:
        self.model = RandomForestClassifier(random_state=conf['general']['random_state'])

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        X = df.drop(columns=['target'])
        y = df['target']
        return train_test_split(X, y, test_size=test_size, random_state=conf['general']['random_state'])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        logging.info("Training the model...")
        self.model.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        logging.info("Testing the model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
        else:
            path = os.path.join(MODEL_DIR, path)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Starting training script...")

    data_proc = DataProcessor()
    trainer = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    trainer.run_training(df, test_size=conf['train']['test_size'])

    logging.info("Training script completed successfully.")


if __name__ == "__main__":
    main()
