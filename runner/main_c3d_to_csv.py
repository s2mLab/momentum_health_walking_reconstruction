from pathlib import Path
import os

import ezc3d
import numpy as np


allow_header_prefix = ["pilot", "MomentumHealth"]
csv_header = [
    "LFHD",
    "RFHD",
    "LBHD",
    "RBHD",
    "C2",
    "C7",
    "T10",
    "CLAV",
    "RBAK",
    "LSHO",
    "LUPA",
    "LELB",
    "LFRM",
    "LWRA",
    "LWRB",
    "LFIN",
    "RSHO",
    "RUPA",
    "RELB",
    "RFRM",
    "RWRA",
    "RWRB",
    "RFIN",
    "LASI",
    "LPSI",
    "RPSI",
    "LTHI",
    "LKNE",
    "LTIB",
    "LANK",
    "LHEE",
    "LTOE",
    "RTHI",
    "RKNE",
    "RTIB",
    "RANK",
    "RHEE",
    "RTOE",
    "PELO",
    "PELA",
    "PELL",
    "PELP",
    "LFEO",
    "LFEA",
    "LFEL",
    "LFEP",
    "LTIO",
    "LTIA",
    "LTIL",
    "LTIP",
    "LFOO",
    "LFOA",
    "LFOL",
    "LFOP",
    "RFEO",
    "RFEA",
    "RFEL",
    "RFEP",
    "RTIO",
    "RTIA",
    "RTIL",
    "RTIP",
    "RFOO",
    "RFOA",
    "RFOL",
    "RFOP",
    "LTOO",
    "LTOA",
    "LTOL",
    "LTOP",
    "RTOO",
    "RTOA",
    "RTOL",
    "RTOP",
    "HEDO",
    "HEDA",
    "HEDL",
    "HEDP",
    "LCLO",
    "LCLA",
    "LCLL",
    "LCLP",
    "RCLO",
    "RCLA",
    "RCLL",
    "RCLP",
    "TRXO",
    "TRXA",
    "TRXL",
    "TRXP",
    "LHUO",
    "LHUA",
    "LHUL",
    "LHUP",
    "LRAO",
    "LRAA",
    "LRAL",
    "LRAP",
    "LHNO",
    "LHNA",
    "LHNL",
    "LHNP",
    "RHUO",
    "RHUA",
    "RHUL",
    "RHUP",
    "RRAO",
    "RRAA",
    "RRAL",
    "RRAP",
    "RHNO",
    "RHNA",
    "RHNL",
    "RHNP",
    "RASI",
    "STRN",
    "LHipAngles",
    "LKneeAngles",
    "LAbsAnkleAngle",
    "LAnkleAngles",
    "RHipAngles",
    "RKneeAngles",
    "RAnkleAngles",
    "RAbsAnkleAngle",
    "LPelvisAngles",
    "RPelvisAngles",
    "LFootProgressAngles",
    "RFootProgressAngles",
    "RNeckAngles",
    "LNeckAngles",
    "RSpineAngles",
    "LSpineAngles",
    "LShoulderAngles",
    "LElbowAngles",
    "LWristAngles",
    "RShoulderAngles",
    "RElbowAngles",
    "RWristAngles",
    "RThoraxAngles",
    "LThoraxAngles",
    "RHeadAngles",
    "LHeadAngles",
]


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))

    # Sucessively read c3d files
    for c3d_file in data_base_folder.glob("*.c3d"):
        print(f"Processing file: {c3d_file}")
        c3d = ezc3d.c3d(str(c3d_file))

        # Extract the point names
        point_labels: list[str] = c3d.parameters.POINT.LABELS["value"]

        # Prepare the output data (4 x N x num_frames)
        out = np.ndarray((4, len(csv_header), c3d["data"]["points"].shape[2]), dtype=float) * np.nan
        for i, name in enumerate(csv_header):
            if name in point_labels:
                out[:, i, :] = c3d["data"]["points"][:, point_labels.index(name), :]
            else:
                for prefix in allow_header_prefix:
                    prefixed_name = f"{prefix}:{name}"
                    if prefixed_name in point_labels:
                        out[:, i, :] = c3d["data"]["points"][:, point_labels.index(prefixed_name), :]
                        break
        out = out[:3, :, :]  # Drop the line of ones

        # Flat the data to (num_frames x (N*3))
        frame_count = out.shape[2]
        out = np.einsum("ijk->kji", out).reshape(frame_count, len(csv_header) * 3)

        # Prepare a header coherent with the flattened data (adding _X, _Y, _Z to each element)
        csv_header_expanded = []
        for name in csv_header:
            csv_header_expanded.append(f"{name}_X")
            csv_header_expanded.append(f"{name}_Y")
            csv_header_expanded.append(f"{name}_Z")

        # Save to csv
        out_folder = c3d_file.parent / "processed"
        out_folder.mkdir(parents=True, exist_ok=True)
        out_filepath = out_folder / c3d_file.with_suffix(".csv").name
        print(f"Saving to file: {out_filepath}")

        header_line = ",".join(csv_header_expanded)
        np.savetxt(out_filepath, out, delimiter=",", header=header_line, comments="")


if __name__ == "__main__":
    main()
