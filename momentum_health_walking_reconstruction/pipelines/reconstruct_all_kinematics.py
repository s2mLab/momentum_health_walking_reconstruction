import os
import logging
from pathlib import Path

from biobuddy import ViewAs

from ..models.lower_body import generate_lower_body_model


def reconstruct_all_kinematics(
    data_base_folder: Path,
    models_base_folder: Path,
    subject_names: list[str],
    results_folder: Path,
    output_model_name: str = "lower_body.bioMod",
    override_existing_model: bool = False,
    animate_models: bool = False,
):
    _logger = logging.getLogger(__name__)

    for subject in subject_names:
        _logger.info(f"Reconstructing kinematics for subject {subject}...")

        data_folder = data_base_folder / subject
