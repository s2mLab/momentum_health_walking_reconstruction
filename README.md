# momentum_health_walking_reconstruction
A collection of scripts to analyse the MomentumHealth data collection project. 

## Starting

### Dependencies

First, install the required dependencies using pip:

```bash
pip install . && pip uninstall -y momentum_health_walking_reconstruction
```
Please note `ezc3d` is not yet available on PyPI for Python 3.14, so if the previous command fails, make sure to downgrade to Python 3.13 or lower.
If you need Python 3.14 installed, you can compile `ezc3d` from the source (https://github.com/pyomeca/ezc3d).


### Using the package

If using `vscode`, copy-paste the `.vscode/launch.json.default` to `.vscode/launch.json` and change the `<Path to your data folder>` to the actual path of the data folder on your computer.

If not using `vscode`, set the environment variable `DATA_BASE_FOLDER` to the path of the data folder on your computer.

## Available scripts

- `runner/main_extract_header_info.py`: Example of how to extract header information from c3d files.
- `runner/main_c3d_to_csv.py`: Convert all c3d files in the data folder to csv files.

