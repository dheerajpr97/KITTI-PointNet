import os

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created.")
    else:
        print(f"Directory {directory} already exists.")

def create_file(file_path, content=""):
    with open(file_path, 'w') as file:
        file.write(content)
        print(f"File {file_path} created.")

# Define the structure of your project here
project_structure = {    
        "data": {
            "kitti_raw": {},
            "processed": {},
            "preprocess_scripts": {}
        },
        "models": {
            "pointnet.py": "# PointNet model architecture",
            "utils.py": "# Utility functions for the model"
        },
        "notebooks": {
            "data_preparation.ipynb": "", 
            "model_training.ipynb":"",
            "model_evaluation.ipynb": ""
        },
        "training": {
            "train.py": "# Main training script",
            "config.py": "# Configuration file for training parameters"
        },
        "evaluation": {
            "evaluate.py": "# Script to evaluate the trained model",
            "metrics.py": "# Custom metrics for model evaluation"
        },
        "requirements.txt": "# Project dependencies",
        ".gitignore": "data/kitti_raw/\ndata/processed/",
        "README.md": "# Project Title\n\n## Overview\n\n## Installation\n\n## Usage",
        "LICENSE": ""
    }


# Function to create the project structure
def create_project_structure(base_path, structure):
    for path, content in structure.items():
        full_path = os.path.join(base_path, path)
        if isinstance(content, dict):
            create_dir(full_path)
            create_project_structure(full_path, content)
        else:
            create_file(full_path, content)

# Create the project structure
create_project_structure(".", project_structure)
