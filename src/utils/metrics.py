
import torch
import torch.nn.functional as F

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


def compute_rate_distortion_loss(out, x, lmbda, image_shape):
    """Compute rate-distortion loss.
    out: output of the model, {'features': {'p2': tensor, 'p3': tensor, 'p4': tensor, 'p5': tensor, 'p6': tensor}, 'likelihoods': {'p2': tensor, 'p3': tensor, 'p4': tensor, 'p5': tensor, 'p6': tensor}}
    x: input features, {'p2': tensor, 'p3': tensor, 'p4': tensor, 'p5': tensor, 'p6': tensor}
    lmbda: lambda for the rate-distortion trade-off
    image_shape: shape of the image, (N, C, H, W)
    """
    N, _, H, W = image_shape
    num_pixels = N * H * W
    weights = {'p2': 0.2, 'p3': 0.2, 'p4': 0.2, 'p5': 0.2, 'p6': 0.2}
    mse_loss = sum(weights[key] * F.mse_loss(x[key], out['features'][key], reduction='mean') for key in x.keys()) 
    # note that bpp is average over original image instead of features, align with MPEG-VCM common test conditions
    bpp_loss = sum(torch.log2(likelihoods).sum() / (-num_pixels)
                  for likelihoods in out['likelihoods'].values())
    loss = lmbda * mse_loss + bpp_loss
    return loss, mse_loss, bpp_loss
