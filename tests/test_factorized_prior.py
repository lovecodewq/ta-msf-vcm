import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from pathlib import Path
import numpy as np
from model.factorized_prior import FactorizedPrior
from data.transforms import create_transforms
import yaml
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model instance
    model = FactorizedPrior(
        n_hidden=config['model']['n_hidden'],
        n_channels=config['model']['n_channels']
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def process_image(image_path, transform):
    """Load and preprocess image."""
    img = Image.open(image_path).convert('RGB')
    x = transform(img)
    return x.unsqueeze(0), img  # Return both tensor and original PIL image

def calculate_bpp(strings, image_shape):
    """Calculate bits per pixel from compressed strings."""
    num_pixels = image_shape[0] * image_shape[1]
    total_bits = sum(len(s) * 8 for s in strings)  # Convert bytes to bits
    return total_bits / num_pixels

def create_comparison_image(original_img, reconstructed_img, bpp):
    """Create a side-by-side comparison with labels."""
    # Get dimensions of original image
    width, height = original_img.size
    
    # Create a new image with space for both images and labels
    margin = 20
    label_height = 30
    total_width = width * 2 + margin * 3
    total_height = height + margin * 2 + label_height
    
    comparison = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste the images
    comparison.paste(original_img, (margin, margin + label_height))
    
    # Resize reconstructed image to match original size if needed
    if reconstructed_img.size != original_img.size:
        reconstructed_img = reconstructed_img.resize(original_img.size, Image.Resampling.LANCZOS)
    comparison.paste(reconstructed_img, (width + margin * 2, margin + label_height))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    try:
        # Try to use a standard font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw labels
    draw.text((margin, margin), "Original", fill='black', font=font)
    draw.text((width + margin * 2, margin), 
              f"Reconstructed (BPP: {bpp:.3f})", 
              fill='black', font=font)
    
    return comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Path to save reconstructed image')
    args = parser.parse_args()

    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load model
    model, config = load_model(args.model, device)
    logging.info('Model loaded successfully')

    # Create transform using test_transforms instead of regular transforms
    transform = create_transforms(config, split='test')

    # Process image
    x, original_img = process_image(args.image, transform)
    x = x.to(device)
    
    # Get original image dimensions for bpp calculation
    original_size = original_img.size

    with torch.no_grad():
        # Compress
        compressed = model.compress(x)
        strings = compressed['y_strings']
        shape = compressed['shape']
        
        # Calculate bpp
        bpp = calculate_bpp(strings, original_size)
        logging.info(f'Compression rate: {bpp:.4f} bits per pixel')

        # Decompress
        decompressed = model.decompress(strings, shape)
        x_hat = decompressed['x_hat']

        # Convert reconstructed tensor to PIL image
        x_hat = x_hat.squeeze(0).cpu().clamp(0, 1)
        to_pil = transforms.ToPILImage()
        reconstructed_img = to_pil(x_hat)
        
        # Create comparison image
        comparison = create_comparison_image(original_img, reconstructed_img, bpp)
        
        # Create output directory if it doesn't exist
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If output is a directory, create a filename
        if output_path.is_dir():
            input_name = Path(args.image).stem
            output_path = output_path / f"{input_name}_comparison.png"
        
        # Save comparison image
        comparison.save(output_path)
        logging.info(f'Comparison image saved to: {output_path}')

if __name__ == '__main__':
    main() 