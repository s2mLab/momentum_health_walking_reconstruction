import os
import logging
from pathlib import Path

from biobuddy import ViewAs

from ..models.lower_body import generate_lower_body_model


def generate_all_models(
    data_base_folder: Path,
    subject_names: list[str],
    results_folder: Path,
    output_model_name: str = "lower_body.bioMod",
    override_existing_models: bool = False,
    animate_models: bool = False,
):
    _logger = logging.getLogger(__name__)

    for subject in subject_names:
        _logger.info(f"Generating model for subject {subject}...")

        calibration_folder = data_base_folder / subject / "calibration_files"
        output_model_filepath = results_folder / subject / output_model_name

        if not override_existing_models and output_model_filepath.exists():
            _logger.info(f"Model file already exists and override is set to False, skipping.")
            continue
        model = generate_lower_body_model(calibration_folder=calibration_folder, use_score=True)

        # Put the model together, print it and print it to a bioMod file
        os.makedirs(output_model_filepath.parent, exist_ok=True)
        model.to_biomod(output_model_filepath)

        if animate_models:
            model.animate(view_as=ViewAs.BIORBD_BIOVIZ, model_path=output_model_filepath.as_posix())
