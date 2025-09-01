# ChemProp Structured Results Management

## Overview

ChemProp now includes built-in structured results management to prevent file conflicts during parallel runs and organize all outputs systematically.

## Key Features

### 1. Automatic Unique Directories
Every training run gets a unique directory with the pattern:
```
results/{dataset_name}/{timestamp}_{mode}_seed{N}/
```

Example:
```
results/freesolv/20250826_234002_cbp_seed42/
```

### 2. No File Conflicts
- Multiple parallel runs with the same seed won't overwrite each other
- Timestamp ensures uniqueness even with identical parameters
- Perfect for Active Learning integration where multiple models train simultaneously

### 3. Organized Structure
```
results/
└── freesolv/
    └── 20250826_234002_cbp_seed42/
        ├── model_0/
        │   ├── model.pt           # Model checkpoint
        │   └── loss_history.json   # Loss with metadata
        ├── cbp_logs/              # CBP logs (CBP mode only)
        │   ├── cbp_training_*.log
        │   └── cbp_training_*_summary.json
        ├── training_summary.json   # Overall summary
        └── test_scores.json       # Test metrics
```

### 4. Enhanced Metadata
Loss history now includes:
- Mode (standard/cbp)
- Seed
- Dataset name
- Timestamp
- Model index

Training summary includes:
- CBP parameters (if CBP mode)
- Complete experiment configuration
- Timestamp of completion

## Usage

### Default Behavior (Structured Directories)
```bash
# Standard training
chemprop_train --data_path data.csv --dataset_type regression --epochs 30

# CBP training
chemprop_train --data_path data.csv --dataset_type regression --epochs 30 --cbp
```

Results will be in: `results/{dataset}/{timestamp}_{mode}_seed{N}/`

### Backward Compatibility Mode
To use a specific directory without automatic structuring:
```bash
chemprop_train --data_path data.csv --save_dir my_custom_dir --no_structured_dir
```

## Integration with Active Learning

This structured approach is designed for seamless Active Learning integration:

1. **No Conflicts**: Multiple acquisition functions can train models in parallel
2. **Easy Tracking**: Each experiment is uniquely identified
3. **Programmatic Access**: JSON metadata makes it easy to find and compare results
4. **CBP Support**: CBP logs are organized within each experiment

## Finding Results

### Latest Experiment
```python
from pathlib import Path
import json

# Find all experiments for a dataset
experiments = Path("results/freesolv").glob("20*")
latest = max(experiments, key=lambda p: p.stat().st_mtime)
print(f"Latest: {latest}")

# Load summary
with open(latest / "training_summary.json") as f:
    summary = json.load(f)
    print(f"RMSE: {summary['ensemble_test_scores']['rmse']}")
```

### Compare CBP vs Standard
```python
# Find all CBP runs
cbp_runs = Path("results/freesolv").glob("*_cbp_*")
for run in cbp_runs:
    with open(run / "training_summary.json") as f:
        summary = json.load(f)
        print(f"{run.name}: {summary['ensemble_test_scores']}")
```

## Implementation Details

### Modified Files
1. **run_training.py**: Added `get_structured_save_dir()` function
2. **cbp_trainer.py**: Uses environment variable for log directory
3. **args.py**: Added `no_structured_dir` flag

### No Core Logic Changes
- Training logic remains unchanged
- Model architecture unchanged
- Loss calculation unchanged
- All existing features preserved

## Environment Variables

- `CBP_LOG_DIR`: Set automatically for CBP runs to organize logs

## Migration Guide

### From Old Structure
```bash
# Old way (files might conflict)
chemprop_train --data_path data.csv --save_dir results

# New way (automatic organization)
chemprop_train --data_path data.csv
# Results in: results/data/20250826_143022_standard_seed0/
```

### For Scripts/Automation
```python
# If you need the exact save directory
args = TrainArgs().parse_args([...])
data = get_data(path=args.data_path, args=args)
scores = run_training(args=args, data=data)
print(f"Results saved to: {args.save_dir}")  # This contains the actual path used
```

## Troubleshooting

### Q: How to disable structured directories?
A: Use `--no_structured_dir` flag

### Q: Where are my results?
A: Check `results/{dataset_name}/` or the path printed during training

### Q: Can I use custom directories?
A: Yes, use `--save_dir custom_path --no_structured_dir`

### Q: How to clean old results?
A: `rm -rf results/` (be careful!)

## Summary

✅ **Integrated**: Works within existing ChemProp workflow
✅ **Safe**: No file conflicts in parallel runs
✅ **Organized**: Clear, searchable structure
✅ **Compatible**: Backward compatible with flag
✅ **Ready**: Perfect for Active Learning integration
