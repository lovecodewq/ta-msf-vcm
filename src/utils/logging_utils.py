"""
Logging utilities for training experiments.
"""

import logging
from pathlib import Path
from datetime import datetime
import yaml


def create_run_directory(base_save_dir, config):
    """Create a unique directory for this training run.
    
    Args:
        base_save_dir (str): Base directory for saving runs
        config (dict): Training configuration dictionary
        
    Returns:
        Path: Path to the created run directory
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract key parameters for directory naming
    lambda_val = config['training']['lambda']
    lr = config['training']['learning_rate']
    batch_size = config['training']['batch_size']
    
    # Create descriptive directory name
    run_name = f"run_{timestamp}_lambda_{lambda_val:.2e}_lr_{lr:.2e}_bs_{batch_size}"
    
    # Create full path
    run_dir = Path(base_save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def setup_logging(save_dir):
    """Setup logging with unique log file name.
    
    Args:
        save_dir (Path): Directory to save log files
    """
    # Create a more descriptive log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f'train_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log the save directory for reference
    logging.info(f"Training logs saved to: {log_file}")
    logging.info(f"Run directory: {save_dir}")


def save_training_metadata(save_dir, config, args):
    """Save training metadata for reproducibility.
    
    Args:
        save_dir (Path): Directory to save metadata
        config (dict): Training configuration
        args: Command line arguments
    """
    # Save configuration file for reproducibility
    config_backup_path = save_dir / 'config_used.yaml'
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Save command line arguments
    args_backup_path = save_dir / 'command_args.txt'
    with open(args_backup_path, 'w') as f:
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")
        f.write(f"Command executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logging.info(f"Configuration saved to: {config_backup_path}")
    logging.info(f"Command arguments saved to: {args_backup_path}")


def setup_training_environment(base_save_dir, config, args):
    """Setup complete training environment with logging and metadata.
    
    Args:
        base_save_dir (str): Base directory for saving runs
        config (dict): Training configuration
        args: Command line arguments
        
    Returns:
        Path: Path to the created run directory
    """
    # Create unique run directory
    save_dir = create_run_directory(base_save_dir, config)
    
    # Setup logging
    setup_logging(save_dir)
    
    # Save metadata for reproducibility
    save_training_metadata(save_dir, config, args)
    
    return save_dir 