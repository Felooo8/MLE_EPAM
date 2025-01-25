import unittest
import pandas as pd
import os
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, "settings.json"))

class DataProcessor:
    @staticmethod
    def load_iris_dataset():
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['target'] = iris.target
        return data

    @staticmethod
    def split_data(data, test_size=0.2, random_state=47):
        return train_test_split(data, test_size=test_size, random_state=random_state)


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(CONF_FILE):
            with open(CONF_FILE, "r") as file:
                conf = json.load(file)
        else:
            conf = {
                "general": {"data_dir": os.path.join(ROOT_DIR, "data")},
                "train": {"table_name": "train.csv"},
                "inference": {"inp_table_name": "inference.csv"}
            }
            with open(CONF_FILE, "w") as file:
                json.dump(conf, file, indent=4)

        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])
        cls.inference_path = os.path.join(cls.data_dir, conf['inference']['inp_table_name'])
        os.makedirs(cls.data_dir, exist_ok=True)

    def test_load_iris_dataset(self):
        dp = DataProcessor()
        data = dp.load_iris_dataset()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('target', data.columns)
        self.assertEqual(data.shape[1], 5)  # 4 features + 1 target

    def test_split_data(self):
        dp = DataProcessor()
        data = dp.load_iris_dataset()
        train_data, inference_data = dp.split_data(data, test_size=0.2, random_state=47)
        self.assertGreater(len(train_data), len(inference_data))
        self.assertEqual(len(train_data) + len(inference_data), len(data))


class TestTraining(unittest.TestCase):
    def test_model_training(self):
        # Simulate a training setup
        data = pd.DataFrame({
            'feature1': [5.1, 4.9, 4.7, 4.6],
            'feature2': [3.5, 3.0, 3.2, 3.1],
            'feature3': [1.4, 1.4, 1.3, 1.5],
            'feature4': [0.2, 0.2, 0.2, 0.2],
            'target': [0, 0, 0, 0]
        })
        X_train = data.drop(columns=['target'])
        y_train = data['target']

        # Mock a simple training class
        class Training:
            def __init__(self):
                self.model = None

            def train(self, X, y):
                from sklearn.tree import DecisionTreeClassifier
                self.model = DecisionTreeClassifier()
                self.model.fit(X, y)

        tr = Training()
        tr.train(X_train, y_train)
        self.assertIsNotNone(tr.model)
        self.assertTrue(hasattr(tr.model, 'predict'))


if __name__ == '__main__':
    unittest.main()
