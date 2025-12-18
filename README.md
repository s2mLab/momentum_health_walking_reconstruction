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

- `main_model_creation.py`: Create personalized kinematic models from static trials.
- `main_kinematics_reconstruction.py`: Reconstruct the kinematics of walking trials using an extended Kalman filter.
- `main_visualize_reconstruction.py`: Visualize the reconstructed kinematics using `pyomeca`.


## Results folder

### Biomechanical model
The biomechanical model is based on a kinematic chain with 24 degrees of freedom (DOF). 
The output file containing the model definition is named `lower_body.bioMod`.

Here is a quick overview of the segments and their respective DOF:

| Name        | Parent      | Translation DoF | Rotation DoF | Axes Directions        |
|-------------|-------------|-----------------|--------------|------------------------|
| Pelvis      | -           | 3 (X, Y, Z)     | 3 (X, Y, Z)  | Y forward, X' right    |
| Trunk       | Pelvis      | 3 (X, Y, Z)     | 3 (X, Y, Z)  | Z up, Y' forward       |
| Left Femur  | Pelvis      | 0               | 3 (X, Z, Y)  | Z up, X' right         |
| Left Tibia  | Left Femur  | 0               | 1 (X)        | X right, Z' up         |
| Left Foot   | Left Tibia  | 0               | 2 (X, Z)     | Z along foot, X' right |
| Right Femur | Pelvis      | 0               | 3 (X, Z, Y)  | Z up, X' right         |
| Right Tibia | Right Femur | 0               | 1 (X)        | X right, Z' up         |
| Right Foot  | Right Tibia | 0               | 2 (X, Z)     | Z along foot, X' right |
    
    Please note that the first axis is determined from the anatomical landmarks, while the second (with a prime) is determined from the anatomical landmarks but orthogonalized with respect to the first axis and third axes. The third axis is determined using the right-hand rule using the first and second axes (before orthogonalization).

Due to the use of the right-hand rule, some rotations are inverted compared to the opposite side. The following table summarizes the definition of positive rotations and translations for each DOF:


| Index | Segment     | DOF           | Definition of positive     |
|-------|-------------|---------------|----------------------------|
| 0     | Pelvis      | Translation X | Translating to the right   |
| 1     | Pelvis      | Translation Y | Translating forward        |
| 2     | Pelvis      | Translation Z | Translating upward         |
| 3     | Pelvis      | Rotation X    | Rotating about X axis      |
| 4     | Pelvis      | Rotation Y    | Rotating about Y axis      |
| 5     | Pelvis      | Rotation Z    | Rotating about Z axis      |
| 6     | Trunk       | Translation X | Translating to the right   |
| 7     | Trunk       | Translation Y | Translating forward        |
| 8     | Trunk       | Translation Z | Translating upward         |
| 9     | Trunk       | Rotation X    | Rotating about X axis      |
| 10    | Trunk       | Rotation Y    | Rotating about Y axis      |
| 11    | Trunk       | Rotation Z    | Rotating about Z axis      |
| 12    | Left Femur  | Rotation X    | Hip flexion                |
| 13    | Left Femur  | Rotation Z    | Hip external rotation      |
| 14    | Left Femur  | Rotation Y    | Hip adduction              |
| 15    | Left Tibia  | Rotation X    | Knee extension             |
| 16    | Left Foot   | Rotation X    | Dorsiflexion               |
| 17    | Left Foot   | Rotation Z    | Inversion                  |
| 18    | Right Femur | Rotation X    | Hip flexion                |
| 19    | Right Femur | Rotation Z    | Hip internal rotation      |
| 20    | Right Femur | Rotation Y    | Hip abduction              |
| 21    | Right Tibia | Rotation X    | Knee extension             |
| 22    | Right Foot  | Rotation X    | Dorsiflexion               |
| 23    | Right Foot  | Rotation Z    | Eversion                   |

### Kinematics reconstruction

The kinematics reconstruction is performed using an extended Kalman filter (EKF) approach.
It consists of finding the optimal set of joint angles that minimize the difference between the experimental marker positions and the model's virtual markers.

The output files containing the reconstructed kinematics are named `<trial_name>_q.npy`.
In these files, `q` represents the joint angles for each degree of freedom (DOF) of the model (first dimension), over time (second dimension).

Therefore, each row corresponds to a specific DOF, which can be mapped to the segments and their respective DOFs as described in the second table of the biomechanical model section above.
