import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration file path
CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, "settings.json"))

# Load configuration settings
logger.info("Loading configuration settings...")
if os.path.exists(CONF_FILE):
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)
else:
    conf = {
        "general": {
            "data_dir": DATA_DIR
        },
        "train": {
            "table_name": "train.csv"
        },
        "inference": {
            "inp_table_name": "inference.csv"
        }
    }
    with open(CONF_FILE, "w") as file:
        json.dump(conf, file, indent=4)

# Define paths
logger.info("Defining paths...")
TRAIN_PATH = os.path.join(conf['general']['data_dir'], conf['train']['table_name'])
INFERENCE_PATH = os.path.join(conf['general']['data_dir'], conf['inference']['inp_table_name'])

# Function to load and process the Iris dataset
def load_and_process_iris(test_size=0.2, random_state=47):
    """
    Loads the Iris dataset, splits it into training and inference datasets,
    and saves them as CSV files.

    Args:
        test_size (float): Proportion of the dataset to include in the inference set.
        random_state (int): Random seed for reproducibility.

    Returns:
        None
    """
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    logger.info("Splitting dataset into training and inference sets...")
    train_data, inference_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    logger.info(f"Saving training data to {TRAIN_PATH}...")
    train_data.to_csv(TRAIN_PATH, index=False)
    logger.info(f"Saving inference data to {INFERENCE_PATH}...")
    inference_data.to_csv(INFERENCE_PATH, index=False)

    logger.info(f"Training data saved to {TRAIN_PATH}")
    logger.info(f"Inference data saved to {INFERENCE_PATH}")

# Main execution
if __name__ == "__main__":
    try:
        load_and_process_iris()
        logger.info("Dataset processing completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
