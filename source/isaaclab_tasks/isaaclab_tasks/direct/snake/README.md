### Have the usd_files directory in the same directory as the snake_env.py file. 
> rl_snake->source->isaaclab_tasks->issaclab_tasks->direct->snake->usd_files
### The usd files are available in the following link: 
#### https://drive.google.com/drive/folders/1rtDEspDEIxF3HscERGqdY5M4fiMFNo0d?usp=drive_link

### For training: in the top directory
```./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Snake-FixedBase-v0 --headless```

### For testing: in the top directory
```./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Snake-FixedBase-v0```

### For tensorboard: in the top directory
```./isaaclab.sh -p -m tensorboard.main --logdir=logs```

### If you want to test the manual control of the snake, set enable_manual_oscillation: bool = True in the snake_env.py file.
### Two modes are available:
#### 1. Sinusoidal oscillation (use: sidewinding)
#### 2. Constant velocity for all joints (use: constant)

### For visualzing the target point, set show_marker: bool = True in the snake_env.py file. Dont set it to True whilr training, as it will slow down the training process.
