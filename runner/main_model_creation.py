"""
This example shows how to create a personalized kinematic model from a C3D file containing a static trial.
Here, we generate a simple lower-body model with only a trunk segment.
The marker position and names are taken from Maldonado & al., 2018 (https://hal.science/hal-01841355/)
"""

import os
from pathlib import Path
import logging

from biobuddy import ViewAs
from momentum_health_walking_reconstruction import generate_lower_body_model


logger = logging.getLogger(__name__)


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load the paths from environment variables
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    results_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES").split(",")
    output_name = "lower_body.bioMod"

    # Load options from environment variables
    override_existing_model = os.getenv("OVERRIDE_EXISTING_MODEL", "false") == "true"
    animate_model = os.getenv("ANIMATE_MODEL", "false") == "true"

    for subject in subject_names:
        logger.info(f"Generating model for subject {subject}...")

        calibration_folder = data_base_folder / subject / "calibration_files"
        output_model_filepath = results_folder / subject / output_name

        if not override_existing_model and output_model_filepath.exists():
            logger.info(f"Model file already exists and override is set to False, skipping.")
            continue
        model = generate_lower_body_model(calibration_folder=calibration_folder, use_score=True)

        # Put the model together, print it and print it to a bioMod file
        os.makedirs(output_model_filepath.parent, exist_ok=True)
        model.to_biomod(output_model_filepath)

        if animate_model:
            model.animate(view_as=ViewAs.BIORBD_BIOVIZ, model_path=output_model_filepath.as_posix())


if __name__ == "__main__":
    main()
