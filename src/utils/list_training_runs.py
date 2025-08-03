#!/usr/bin/env python3
"""
Utility script to list and manage training runs for the image compression project.
"""

import os
import sys
from pathlib import Path
import yaml
from datetime import datetime
import argparse

def get_project_root():
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # This script is in src/utils/ directory, so go up two levels
    return current_file.parent.parent.parent

def list_runs(base_dir, run_type="all"):
    """List all training runs in the specified directory."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist.")
        return
    
    # Find all run directories
    run_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('run_'):
            run_dirs.append(item)
    
    if not run_dirs:
        print(f"No training runs found in {base_dir}")
        return
    
    # Sort by creation time (newest first)
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n{'='*80}")
    print(f"Training Runs in {base_dir}")
    print(f"{'='*80}")
    print(f"{'Index':<5} {'Run Name':<40} {'Status':<10} {'Best Loss':<12} {'Last Modified'}")
    print(f"{'-'*80}")
    
    for idx, run_dir in enumerate(run_dirs, 1):
        # Extract info from run directory
        run_name = run_dir.name
        last_modified = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        
        # Check if training completed
        best_model_path = run_dir / 'best_model.pth'
        config_path = run_dir / 'config_used.yaml'
        
        status = "Unknown"
        best_loss = "N/A"
        
        if best_model_path.exists():
            try:
                import torch
                checkpoint = torch.load(best_model_path, map_location='cpu')
                best_loss = f"{checkpoint.get('best_loss', 'N/A'):.4f}"
                status = "Completed"
            except:
                status = "Error"
        elif config_path.exists():
            status = "Running/Failed"
        
        print(f"{idx:<5} {run_name:<40} {status:<10} {best_loss:<12} {last_modified}")
    
    print(f"{'-'*80}")
    print(f"Total runs: {len(run_dirs)}")

def show_run_details(run_dir):
    """Show detailed information about a specific run."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"Run directory {run_dir} does not exist.")
        return
    
    print(f"\n{'='*60}")
    print(f"Run Details: {run_path.name}")
    print(f"{'='*60}")
    
    # Show config
    config_path = run_path / 'config_used.yaml'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"\nConfiguration:")
            print(f"  Model: {config.get('model', {}).get('name', 'N/A')}")
            print(f"  Lambda: {config.get('training', {}).get('lambda', 'N/A')}")
            print(f"  Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
            print(f"  Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}")
            print(f"  Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
        except Exception as e:
            print(f"  Could not read config: {e}")
    
    # Show command args
    args_path = run_path / 'command_args.txt'
    if args_path.exists():
        try:
            with open(args_path, 'r') as f:
                args_content = f.read()
            print(f"\nCommand Arguments:")
            print(f"  {args_content.strip()}")
        except Exception as e:
            print(f"  Could not read command args: {e}")
    
    # Show model checkpoint info
    best_model_path = run_path / 'best_model.pth'
    if best_model_path.exists():
        try:
            import torch
            checkpoint = torch.load(best_model_path, map_location='cpu')
            print(f"\nModel Checkpoint:")
            print(f"  Best Loss: {checkpoint.get('best_loss', 'N/A'):.6f}")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
        except Exception as e:
            print(f"  Could not read checkpoint: {e}")
    
    # Show log files
    log_files = list(run_path.glob('train_*.log'))
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        print(f"\nLatest Log File: {latest_log.name}")
        print(f"  Size: {latest_log.stat().st_size / 1024:.1f} KB")
        print(f"  Last Modified: {datetime.fromtimestamp(latest_log.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List all files in the run directory
    print(f"\nFiles in run directory:")
    for file_path in sorted(run_path.iterdir()):
        if file_path.is_file():
            size_kb = file_path.stat().st_size / 1024
            print(f"  {file_path.name:<30} {size_kb:>8.1f} KB")

def main():
    parser = argparse.ArgumentParser(description="List and manage training runs")
    parser.add_argument('--base-dir', type=str, 
                       help='Base directory to search for runs (default: checkpoints/fpn_compression)')
    parser.add_argument('--show-details', type=str, metavar='RUN_DIR',
                       help='Show detailed information about a specific run directory')
    parser.add_argument('--run-type', choices=['all', 'fpn', 'detection', 'factorized'],
                       default='all', help='Type of runs to show')
    
    args = parser.parse_args()
    
    # Set default base directory
    if args.base_dir is None:
        project_root = get_project_root()
        args.base_dir = project_root / 'checkpoints' / 'fpn_compression'
    
    if args.show_details:
        show_run_details(args.show_details)
    else:
        list_runs(args.base_dir, args.run_type)

if __name__ == "__main__":
    main() 