## IsaacLab Snake Task

### Directory Structure

Ensure the `usd_files` directory is in the same directory as `snake_env.py`:
```
rl_snake/
└── source/
    └── isaaclab_tasks/
        └── isaaclab_tasks/
            └── direct/
                └── snake/
                    ├── snake_env.py
                    └── usd_files/
```

### USD Files

Download the required USD files from the following link and place them in the `usd_files` directory:
- [USD Files Google Drive](https://drive.google.com/drive/folders/1rtDEspDEIxF3HscERGqdY5M4fiMFNo0d?usp=drive_link)

---

## Usage

### Training

From the top-level directory, run:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Snake-FixedBase-v0 --headless
```

### Testing

From the top-level directory, run:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Snake-FixedBase-v0
```

### TensorBoard

To visualize logs with TensorBoard:
```bash
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```

---

## Manual Control

To test manual control of the snake, set `enable_manual_oscillation: bool = True` in `snake_env.py`.

Two modes are available:
1. **Sinusoidal oscillation** (sidewinding)
2. **Constant velocity** for all joints (constant)

---

## Visualizing the Target Point

To visualize the target point, set `show_marker: bool = True` in `snake_env.py`.

> **Note:** Do **not** enable this during training, as it will slow down the process.

---

## Pre-commit Hooks

To ensure code quality and consistency, set up pre-commit hooks:

1. Install [pre-commit](https://pre-commit.com/) if not already installed:
    ```bash
    pip install pre-commit
    ```

2. In the top-level directory (where `.pre-commit-config.yaml` is located), run:
    ```bash
    pre-commit install
    ```

3. Now, pre-commit hooks will run automatically on every commit.

To manually run all hooks on all files:
```bash
pre-commit run --all-files
```
