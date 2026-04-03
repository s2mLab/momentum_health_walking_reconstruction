import json
import logging
from pathlib import Path
import os

import matplotlib.pyplot as plt
from momentum_health_walking_reconstruction import TrialType, Metrics
import numpy as np
from scipy import stats as scipy_stats
import spm1d


page_break = '\n<div style="page-break-after: always;"></div>\n\n'


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))
    model_base_folder = Path(os.getenv("MODELS_BASE_FOLDER"))
    kinematics_base_folder = Path(os.getenv("RESULTS_BASE_FOLDER"))
    subject_names = os.getenv("SUBJECT_NAMES", "").split(",")
    data_matching = json.load(open(os.getenv("DATA_MATCHING_JSON")))
    scalar_stats_to_compare = os.getenv("SCALAR_STATS_TO_COMPARE", "").split(",")
    time_series_list_stats_to_compare_tp = os.getenv("TIME_SERIES_LIST_STATS_TO_COMPARE")
    save_folder = Path(os.getenv("SAVE_FOLDER"))

    if len(scalar_stats_to_compare) < 2:
        raise ValueError(
            "At least two data types must be specified in the 'SCALAR_STATS_TO_COMPARE' environment variable, separated by commas."
        )

    if time_series_list_stats_to_compare_tp is None:
        raise ValueError(
            "At least two data types must be specified in the 'TIME_SERIES_LIST_STATS_TO_COMPARE' environment variable, separated by commas."
        )

    time_series_list_stats_to_compare = []
    for comparison in time_series_list_stats_to_compare_tp.split(";"):
        data_types = comparison.strip("[]").split(",")
        if len(data_types) != 2:
            raise ValueError(
                f"Each comparison in 'TIME_SERIES_LIST_STATS_TO_COMPARE' must contain exactly two data types, separated by a comma. Invalid comparison: '{comparison}'"
            )
        time_series_list_stats_to_compare.append(data_types)

    if save_folder is None:
        raise ValueError("Environment variable 'SAVE_FOLDER' is not set.")

    trial_category = os.getenv("TRIAL_TYPE")
    if trial_category is None:
        raise ValueError(f"Environment variable 'TRIAL_TYPE' is not set.")

    trial_type = data_matching["trial_types"][trial_category]["category"]
    if trial_type == "gait":
        trial_type = TrialType.GAIT
    elif trial_type == "sway":
        trial_type = TrialType.SWAY
    else:
        raise ValueError(f"Could not determine trial type from file path: {trial_category}")

    trial_names = data_matching["trial_types"][trial_category]["trial_names"]
    if not isinstance(trial_names, list):
        raise ValueError(
            f"Expected 'trial_names' to be a list in the data matching JSON for trial category '{trial_category}'."
        )

    save_folder = save_folder / trial_type.name.lower()
    save_folder.mkdir(parents=True, exist_ok=True)
    figure_save_folder = save_folder / "figures"
    figure_save_folder.mkdir(parents=True, exist_ok=True)

    metrics, angles = Metrics.get_mean_metrics(
        data_base_folder=data_base_folder,
        model_base_folder=model_base_folder,
        kinematics_base_folder=kinematics_base_folder,
        subjects=subject_names,
        trial_type=trial_type,
        trial_names=trial_names,
        data_matching=data_matching["data"],
        show_plot=False,
    )

    # Show metrics
    with open(save_folder / f"Report for {trial_type.name.title()} Trials.md", "w") as f:
        f.write(f"# Data Comparison for {trial_type.name.title()} Trials\n\n")
        for metric_type, data in metrics.items():
            f.write(f"## {metric_type.value.replace('_', ' ').title()}\n")
            f.write("### Descriptive Statistics\n")

            used_data_types = []
            data_stats = [np.ndarray((0,)), np.ndarray((0,))]  # data, data_type_index
            f.write(f"| Data Type | Mean | Standard Deviation |\n")
            f.write(f"|-----------|------|-----------------|\n")
            for data_type_index, (data_type, data_by_subjects) in enumerate(data.items()):
                mean = np.nanmean(list(data_by_subjects.values()))
                std = np.nanstd(list(data_by_subjects.values()))
                f.write(f"| {data_type.replace('_', ' ').title()} | {mean:.2f} | {std:.2f} |\n")

                if data_type not in scalar_stats_to_compare:
                    continue
                used_data_types.append(data_type_index)
                for subject, subject_data in data_by_subjects.items():
                    subject_data = np.array([subject_data])
                    data_stats[0] = np.concatenate((data_stats[0], subject_data), axis=0)
                    data_stats[1] = np.concatenate(
                        (data_stats[1], data_type_index * np.ones(len(subject_data))), axis=0
                    )
            f.write(f"\n")
            f.write(f"### Statistical Analysis\n")

            if len(used_data_types) < 2:
                logging.warning(
                    f"Warning: Not enough data types with valid data for {metric_type.value} to perform ANOVA analysis. "
                    "Skipping ANOVA results for this metric."
                )
            else:
                _, p_value = scipy_stats.f_oneway(
                    *[data_stats[0][data_stats[1] == data_type] for data_type in used_data_types]
                )

                f.write("| Comparison | Test | p-value |\n")
                f.write("|------------|------|--------:|\n")

                # ANOVA row
                if len(used_data_types) > 2:
                    f.write(f"| {', '.join(scalar_stats_to_compare)} | ANOVA | {p_value:.4f} |\n")

                # Post-hoc rows
                if p_value < 0.05 or len(used_data_types) == 2:
                    for i in range(len(used_data_types)):
                        for j in range(i + 1, len(used_data_types)):
                            index_i = data_stats[1] == used_data_types[i]
                            index_j = data_stats[1] == used_data_types[j]

                            _, post_hoc_p_value = scipy_stats.ttest_ind(data_stats[0][index_i], data_stats[0][index_j])

                            f.write(
                                f"| {scalar_stats_to_compare[i]} vs {scalar_stats_to_compare[j]} | t-test | {post_hoc_p_value:.4f} |\n"
                            )
            f.write(page_break)

        f.write(f"# Kinematics Comparison\n\n")
        data_by_joints = angles
        for joint, data_by_sides in data_by_joints.items():
            for time_series_stats_to_compare in time_series_list_stats_to_compare:
                for side_index, (side, data_by_types) in enumerate(data_by_sides.items()):
                    if side_index == 0:
                        # Make n-side rows on the figure if there are n sides to compare, and share the x-axis
                        if len(data_by_sides) > 1:
                            fig, axes = plt.subplots(nrows=len(data_by_sides), ncols=1, sharex=True)
                        else:
                            fig = plt.figure()
                        title = f"{joint.name.replace('_', ' ').title()} angles comparison ({' / '.join(time_series_stats_to_compare)})"
                        fig.canvas.manager.set_window_title(title)
                        fig.suptitle(title)
                        fig.set_size_inches(10, 4 * len(data_by_sides))

                    # Select the subplot according to the side index if there are multiple sides, otherwise use the single figure
                    if len(data_by_sides) > 1:
                        plt.sca(axes[side_index])

                    # Create a right-aligned subplot for the SPM results
                    ax_data = plt.gca()
                    ax_spm = axes[side_index].twinx() if len(data_by_sides) > 1 else ax_data.twinx()

                    # Make main axis front but background transparent so it doesn't cover the twinx plot
                    ax_data.patch.set_visible(False)
                    ax_data.set_zorder(ax_spm.get_zorder() + 1)

                    # Compute the stats
                    stats = [
                        np.ndarray((0, 1000)),
                        np.ndarray((0,)),
                        np.ndarray((0,)),
                    ]  # data, data_type_index, subject_index

                    used_data_types = []
                    for data_type_index, (data_type, data_by_subjects) in enumerate(data_by_types.items()):
                        if data_type not in time_series_stats_to_compare:
                            continue
                        used_data_types.append(data_type_index)

                        for subject_index, (subject, trials_data) in enumerate(data_by_subjects.items()):
                            data = np.nanmean(np.array(trials_data), axis=0).mean(axis=0)[None, :]
                            if not data.shape:
                                logging.warning(
                                    f"Warning: Data for {data_type} could not be averaged across trials for subject {subject}. Skipping this subject."
                                )
                                continue
                            stats[0] = np.concatenate((stats[0], data), axis=0)
                            stats[1] = np.concatenate((stats[1], data_type_index * np.ones(data.shape[0])), axis=0)
                            stats[2] = np.concatenate((stats[2], subject_index * np.ones(data.shape[0])), axis=0)

                    if len(used_data_types) < 2:
                        raise RuntimeError(
                            f"Warning: Not enough data types with valid data for {joint.name} {side.name} to perform SPM analysis. Skipping SPM plot for this joint and side."
                        )
                    elif len(used_data_types) == 2:
                        index1 = stats[1] == used_data_types[0]
                        index2 = stats[1] == used_data_types[1]
                        spm = spm1d.stats.ttest(stats[0][index1, :] - stats[0][index2, :])
                        spmi = spm.inference(alpha=0.05, two_tailed=True, interp=True)
                    else:
                        # Repeated-measured not available because of unbalanced data
                        spm = spm1d.stats.anova1(stats[0], stats[1])
                        spmi = spm.inference(0.05)
                    spmi.plot(ax=ax_spm)
                    plt.sca(ax_data)

                    # Plot the data
                    for data_type, data_by_subjects in data_by_types.items():
                        if data_type not in time_series_stats_to_compare:
                            continue
                        all_subjects_mean_data = []
                        for subject, trials_data in data_by_subjects.items():
                            data = np.nanmean(np.array(trials_data), axis=1).mean(axis=0)
                            if not data.shape:
                                logging.warning(
                                    f"Warning: Data for {data_type} could not be averaged across trials for subject {subject}. Skipping this subject."
                                )
                                continue
                            all_subjects_mean_data.append(data)
                        all_subjects_mean_data = np.array(all_subjects_mean_data)
                        mean = np.nanmean(all_subjects_mean_data, axis=0)
                        std = np.nanstd(all_subjects_mean_data, axis=0)
                        time = np.linspace(0, 1000, mean.shape[0])
                        plt.plot(time, mean, label=f"{data_type.replace('_', ' ').title()}")
                        plt.fill_between(time, mean - std, mean + std, alpha=0.2)
                    if len(data_by_sides) > 1:
                        plt.title(f"{side.name.title()}")
                    plt.xlabel("Gait cycle (%)")
                    plt.ylabel("Angle (degrees)")
                    if side_index == len(data_by_sides) - 1:
                        plt.legend()

                # Save the figure
                figure_name = f"{joint.name.replace(' ', '__')}_{side.name}_{'_'.join([str(stat).replace(' ', '__') for stat in time_series_stats_to_compare])}_comparison.png"
                figure_save_path = figure_save_folder / figure_name
                plt.savefig(figure_save_path)

                # Add the figure to the markdown file
                f.write(f"## {joint.name.replace('_', ' ').title()} Comparison\n")
                f.write(
                    f"![{joint.name.replace('_', ' ').title()} comparison]({figure_save_path.relative_to(save_folder)})\n"
                )
                # Skip to next page
                f.write(page_break)

    logging.info("Done.")


if __name__ == "__main__":
    main()
