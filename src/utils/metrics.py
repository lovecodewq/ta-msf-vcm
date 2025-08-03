"""
Utilities for tracking and computing metrics during training.
"""

class AverageMeter:
    """Computes and stores the average and current value.
    
    Commonly used for tracking metrics during training, such as loss values,
    accuracy, etc.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update statistics with new value.
        
        Args:
            val: Value to add to statistics
            n: Number of items this value represents (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 