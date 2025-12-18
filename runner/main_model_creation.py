import os
from pathlib import Path
import logging

from momentum_health_walking_reconstruction import generate_all_models


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    generate_all_models(
        data_base_folder=Path(os.getenv("DATA_BASE_FOLDER")),
        subject_names=os.getenv("SUBJECT_NAMES").split(","),
        results_folder=Path(os.getenv("RESULTS_BASE_FOLDER")),
        override_existing_models=os.getenv("OVERRIDE_EXISTING_MODELS") == "true",
        animate_models=os.getenv("ANIMATE_MODELS") == "true",
    )


if __name__ == "__main__":
    main()
