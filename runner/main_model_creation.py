"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os
from pathlib import Path
import logging

from momentum_health_walking_reconstruction import generate_all_models


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    generate_all_models(
        data_base_folder=Path(os.getenv("DATA_BASE_FOLDER")),
        subject_names=os.getenv("SUBJECT_NAMES").split(","),
        results_folder=Path(os.getenv("RESULTS_BASE_FOLDER")),
        override_existing_models=os.getenv("OVERRIDE_EXISTING_MODELS") == "true",
        animate_models=os.getenv("ANIMATE_MODELS") == "true",
    )


if __name__ == "__main__":
    main()
