# Module 8: Basics of MLE

This repository contains a Data Science project designed as a part of Module 8: Basics of MLE. The project demonstrates a complete Machine Learning Engineering pipeline for working with the Iris flower dataset, including data preparation, model training, and inference, all containerized and executed within Docker environments.

# Project Structure

```
Iris_DS_Project
├── data # Training and inference data
│ ├── iris_train.csv # Training dataset
│ └── iris_inference.csv # Inference dataset
├── models # Trained models
│ └── iris_model.pickle
├── results # Inference results
│ └── predictions.csv
├── training # Scripts and Dockerfile for training
│ ├── train.py # Training script
│ └── Dockerfile # Training Dockerfile
├── inference # Scripts and Dockerfile for inference
│ ├── run.py # Inference script
│ └── Dockerfile # Inference Dockerfile
├── utils.py # Utility functions
├── settings.json # Configuration file
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

## Settings:

The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:

For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:

The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker:

- Build the training Docker image. If the built is successfully done, it will automatically train the model:

```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```

- You may run the container with the following parameters to ensure that the trained model is here:

```bash
docker run -it training_image /bin/bash
```

Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:

```bash
docker cp <container_id>:/app/models/<model_name>.pickle ./models
```

Replace `<container_id>` with your running Docker container ID and `<model_name>.pickle` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:

Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:

```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name> --build-arg settings_name=settings.json -t inference_image .
```

- Run the inference Docker container:

```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```

- Or you may run it with the attached terminal using the following command:

```bash
docker run -it inference_image /bin/bash
```

After that ensure that you have your results in the `results` directory in your inference container.

2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.
