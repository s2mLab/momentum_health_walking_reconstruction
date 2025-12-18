"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os
from pathlib import Path
import logging

from momentum_health_walking_reconstruction import visualize_all_kinematics


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    visualize_all_kinematics(
        data_base_folder=Path(os.getenv("DATA_BASE_FOLDER")),
        models_base_folder=Path(os.getenv("MODELS_BASE_FOLDER")),
        subject_names=os.getenv("SUBJECT_NAMES").split(","),
        results_folder=Path(os.getenv("RESULTS_BASE_FOLDER")),
    )


if __name__ == "__main__":
    main()
