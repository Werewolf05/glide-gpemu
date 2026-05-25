jn  m, ## Profile Model GPU Memory Usage

This folder contains the code to profile the GPU memory usage during model training.

### Prerequisite

1. Install the required packages:
```
pip install pandas matplotlib pycuda
```

2. (Optional) Install transformers for speech models:
```
pip install transformers
```

### Folder Structure

- `memory_profiler.py`:
  - Profiles the peak and persistent memory usage during training for a certain model and batch size.
  - With `--plot-time-series` option, it plots the time series memory usage change during training.

- `utils`
  - `memory.py`: Helper functions to get GPU memory usage.
  - `plot.py`: Helper functions to plot memory usage changes over time.

- `profile_all.py`: Profiles the memory usage for all models and batch sizes.

### Usage

1. Profile the memory usage for a certain model and batch size:

```
python memory_profiler.py -a resnet50 --batch-size 64
```

2. Profile the memory usage for all models and batch sizes:

```
python profile_all.py
```

3. Plot the time series memory usage change during training:

```
python memory_profiler.py -a resnet50 --batch-size 64 --plot-time-series
```
