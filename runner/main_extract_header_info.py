from pathlib import Path
import os

import ezc3d


def main():
    data_base_folder = Path(os.getenv("DATA_BASE_FOLDER"))

    # Sucessively read c3d files
    print(f"The first frame indices for:")
    for c3d_file in data_base_folder.glob("*.c3d"):
        c3d = ezc3d.c3d(str(c3d_file))

        # Extract the header
        first_frame_index: list[str] = c3d.header["points"]["first_frame"]
        print(f"  {c3d_file.name.ljust(40)}: {first_frame_index}")


if __name__ == "__main__":
    main()
