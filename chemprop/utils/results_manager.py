"""
Results Manager for ChemProp
Provides structured result storage with automatic naming to avoid conflicts.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import hashlib


class ResultsManager:
    """
    Manages structured storage of training results, logs, and checkpoints.
    
    Directory structure:
    results/
    ├── {dataset_name}/
    │   ├── {timestamp}_{mode}_{seed}/
    │   │   ├── logs/
    │   │   │   ├── training.log
    │   │   │   ├── cbp_stats.log
    │   │   │   └── loss_history.json
    │   │   ├── checkpoints/
    │   │   │   ├── model_0/
    │   │   │   └── model_1/
    │   │   ├── predictions/
    │   │   │   └── test_preds.csv
    │   │   └── summary.json
    """
    
    def __init__(self, 
                 base_dir: str = "results",
                 dataset_name: str = "default",
                 mode: str = "standard",
                 seed: int = 0,
                 experiment_id: Optional[str] = None):
        """
        Initialize ResultsManager.
        
        :param base_dir: Base directory for all results
        :param dataset_name: Name of the dataset
        :param mode: Training mode ('standard' or 'cbp')
        :param seed: Random seed used
        :param experiment_id: Optional custom experiment ID
        """
        self.base_dir = Path(base_dir)
        self.dataset_name = dataset_name
        self.mode = mode
        self.seed = seed
        
        # Create unique experiment ID if not provided
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"{timestamp}_{mode}_seed{seed}"
        else:
            self.experiment_id = experiment_id
        
        # Set up directory structure
        self.experiment_dir = self.base_dir / dataset_name / self.experiment_id
        self.logs_dir = self.experiment_dir / "logs"
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.predictions_dir = self.experiment_dir / "predictions"
        
        # Create directories
        self._create_directories()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Store metadata
        self.metadata = {
            "experiment_id": self.experiment_id,
            "dataset_name": dataset_name,
            "mode": mode,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "base_dir": str(self.base_dir),
            "experiment_dir": str(self.experiment_dir)
        }
        self._save_metadata()
    
    def _create_directories(self):
        """Create necessary directory structure."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_file = self.logs_dir / "training.log"
        
        # Create a logger with the experiment ID
        logger_name = f"chemprop_{self.experiment_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            # File handler for all logs
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            
            # Console handler for important logs
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger
    
    def _save_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_checkpoint_dir(self, model_idx: int = 0) -> Path:
        """Get directory for a specific model checkpoint."""
        model_dir = self.checkpoints_dir / f"model_{model_idx}"
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    def save_loss_history(self, loss_history: list, model_idx: int = 0, 
                         epoch_details: Dict[int, Dict] = None):
        """
        Save loss history with detailed information.
        
        :param loss_history: List of average losses per epoch
        :param model_idx: Model index in ensemble
        :param epoch_details: Optional detailed stats per epoch
        """
        loss_data = {
            "epochs": list(range(len(loss_history))),
            "losses": loss_history,
            "model_idx": model_idx,
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if epoch_details:
            loss_data["epoch_details"] = epoch_details
        
        # Save to logs directory
        loss_file = self.logs_dir / f"loss_history_model_{model_idx}.json"
        with open(loss_file, 'w') as f:
            json.dump(loss_data, f, indent=2)
        
        # Also log to training log
        self.logger.info(f"Loss history saved for model {model_idx}")
        self.logger.debug(f"Final loss: {loss_history[-1] if loss_history else 'N/A'}")
        
        return loss_file
    
    def save_cbp_stats(self, cbp_stats: Dict, model_idx: int = 0):
        """Save CBP-specific statistics."""
        cbp_file = self.logs_dir / f"cbp_stats_model_{model_idx}.json"
        
        # Add metadata
        cbp_stats["experiment_id"] = self.experiment_id
        cbp_stats["model_idx"] = model_idx
        cbp_stats["timestamp"] = datetime.now().isoformat()
        
        with open(cbp_file, 'w') as f:
            json.dump(cbp_stats, f, indent=2)
        
        # Log summary
        if "replacements_by_epoch" in cbp_stats:
            total_replacements = sum(cbp_stats["replacements_by_epoch"].values())
            self.logger.info(f"CBP: Total neurons replaced: {total_replacements}")
        
        return cbp_file
    
    def save_training_summary(self, summary: Dict):
        """Save overall training summary."""
        summary_file = self.experiment_dir / "summary.json"
        
        # Add metadata
        summary["experiment_metadata"] = self.metadata
        summary["completion_time"] = datetime.now().isoformat()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log key results
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Mode: {self.mode}")
        
        if "ensemble_test_scores" in summary:
            for metric, scores in summary["ensemble_test_scores"].items():
                if scores:
                    avg_score = sum(s for s in scores if s is not None) / len(scores)
                    self.logger.info(f"Test {metric}: {avg_score:.4f}")
        
        self.logger.info(f"Results saved to: {self.experiment_dir}")
        self.logger.info("=" * 60)
        
        return summary_file
    
    def log_epoch(self, epoch: int, loss: float, val_score: float = None, 
                 cbp_replacements: int = None, **kwargs):
        """
        Log information for a single epoch.
        
        :param epoch: Epoch number
        :param loss: Average training loss
        :param val_score: Validation score
        :param cbp_replacements: Number of CBP replacements this epoch
        :param kwargs: Additional metrics to log
        """
        log_msg = f"Epoch {epoch:3d} | Loss: {loss:.6f}"
        
        if val_score is not None:
            log_msg += f" | Val Score: {val_score:.6f}"
        
        if cbp_replacements is not None and cbp_replacements > 0:
            log_msg += f" | CBP Replacements: {cbp_replacements}"
        
        for key, value in kwargs.items():
            if value is not None:
                log_msg += f" | {key}: {value}"
        
        self.logger.info(log_msg)
    
    def get_results_path(self) -> Path:
        """Get the path to the experiment results directory."""
        return self.experiment_dir
    
    def create_unified_log(self):
        """Create a unified log combining all log sources."""
        unified_log_file = self.experiment_dir / "unified_log.txt"
        
        with open(unified_log_file, 'w') as unified:
            unified.write(f"{'='*60}\n")
            unified.write(f"Unified Training Log\n")
            unified.write(f"Experiment: {self.experiment_id}\n")
            unified.write(f"{'='*60}\n\n")
            
            # Add training log
            training_log = self.logs_dir / "training.log"
            if training_log.exists():
                unified.write("=== Training Log ===\n")
                with open(training_log, 'r') as f:
                    unified.write(f.read())
                unified.write("\n\n")
            
            # Add loss history
            for loss_file in sorted(self.logs_dir.glob("loss_history_*.json")):
                unified.write(f"=== {loss_file.name} ===\n")
                with open(loss_file, 'r') as f:
                    data = json.load(f)
                    unified.write(json.dumps(data, indent=2))
                unified.write("\n\n")
            
            # Add CBP stats
            for cbp_file in sorted(self.logs_dir.glob("cbp_stats_*.json")):
                unified.write(f"=== {cbp_file.name} ===\n")
                with open(cbp_file, 'r') as f:
                    data = json.load(f)
                    unified.write(json.dumps(data, indent=2))
                unified.write("\n\n")
            
            # Add summary
            summary_file = self.experiment_dir / "summary.json"
            if summary_file.exists():
                unified.write("=== Summary ===\n")
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    unified.write(json.dumps(data, indent=2))
        
        self.logger.info(f"Unified log created: {unified_log_file}")
        return unified_log_file


def create_results_manager(args) -> ResultsManager:
    """
    Create a ResultsManager from training arguments.
    
    :param args: Training arguments
    :return: Configured ResultsManager instance
    """
    # Extract dataset name from data path
    if hasattr(args, 'data_path') and args.data_path:
        dataset_name = Path(args.data_path).stem
    else:
        dataset_name = "default"
    
    # Determine mode
    mode = "cbp" if hasattr(args, 'cbp') and args.cbp else "standard"
    
    # Get seed
    seed = args.seed if hasattr(args, 'seed') else 0
    
    # Use custom save_dir if provided, otherwise use default
    base_dir = args.save_dir if hasattr(args, 'save_dir') and args.save_dir else "results"
    
    return ResultsManager(
        base_dir=base_dir,
        dataset_name=dataset_name,
        mode=mode,
        seed=seed
    )
