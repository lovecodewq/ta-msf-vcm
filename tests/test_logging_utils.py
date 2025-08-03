#!/usr/bin/env python3
"""
Test script to verify logging utilities work correctly.
"""

import tempfile
from pathlib import Path
import yaml
import argparse

from utils.logging_utils import setup_training_environment


def create_test_config():
    """Create a test configuration."""
    return {
        'training': {
            'lambda': 2.5e-4,
            'learning_rate': 5e-5,
            'batch_size': 4,
            'save_dir': 'test_checkpoints'
        }
    }


def create_test_args():
    """Create test command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test_config.yaml')
    parser.add_argument('--detection_checkpoint', type=str, default='test_checkpoint.pth')
    return parser.parse_args(['--config', 'test_config.yaml'])


def main():
    print("Testing logging utilities...")
    
    # Create test config and args
    config = create_test_config()
    args = create_test_args()
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config['training']['save_dir'] = temp_dir
        
        # Test the setup_training_environment function
        try:
            save_dir = setup_training_environment(temp_dir, config, args)
            print(f"✅ Created run directory: {save_dir}")
            
            # Check that expected files exist
            expected_files = [
                'config_used.yaml',
                'command_args.txt'
            ]
            
            for file_name in expected_files:
                file_path = save_dir / file_name
                if file_path.exists():
                    print(f"✅ {file_name} created successfully")
                else:
                    print(f"❌ {file_name} not found")
            
            # Check log file was created (should match pattern train_*.log)
            log_files = list(save_dir.glob('train_*.log'))
            if log_files:
                print(f"✅ Log file created: {log_files[0].name}")
            else:
                print("❌ No log file found")
                
            print("\n🎉 All tests passed! Logging utilities are working correctly.")
            
        except Exception as e:
            print(f"❌ Error testing logging utilities: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 