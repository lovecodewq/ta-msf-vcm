import argparse
import logging
from pathlib import Path
import struct
from tqdm import tqdm
import torch

from model.joint_autoregress_fpn_compressor import JointAutoregressFPNCompressor


def parse_bitstream(path: Path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'FPN1':
            raise ValueError(f'Invalid magic in {path}, expected FPN1, got {magic!r}')
        header = f.read(16)
        p2_h, p2_w, len_y, len_z = struct.unpack('<IIII', header)
        y_bytes = f.read(len_y)
        z_bytes = f.read(len_z)
    return p2_h, p2_w, y_bytes, z_bytes


def main():
    parser = argparse.ArgumentParser(description='Decompress feature bitstreams and save reconstructed FPN features as .pt')
    parser.add_argument('--bitstreams_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Load compressor
    ckpt = torch.load(args.checkpoint, map_location=device)
    comp_model = JointAutoregressFPNCompressor(
        N=ckpt['config']['model']['n_latent'],
        M=ckpt['config']['model']['n_hidden']
    ).to(device)
    comp_model.load_state_dict(ckpt['model_state_dict'])
    comp_model.eval()
    # Build entropy CDF tables for compression
    try:
        comp_model.update()
    except Exception:
        comp_model.update(force=True)

    bit_dir = Path(args.bitstreams_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_files = sorted(bit_dir.glob('*.bin'))
    if not bin_files:
        logging.warning(f'No .bin files found in {bit_dir}')
        return

    with torch.no_grad():
        for bin_path in tqdm(bin_files, desc='Decompressing features'):
            image_id = bin_path.stem
            p2_h, p2_w, y_bytes, z_bytes = parse_bitstream(bin_path)
            decomp = comp_model.decompress([[y_bytes], [z_bytes]], p2_h, p2_w)
            feats_list = decomp['features']  # list [p2..p6]
            single_feats = {
                'p2': feats_list[0].cpu(),
                'p3': feats_list[1].cpu(),
                'p4': feats_list[2].cpu(),
                'p5': feats_list[3].cpu(),
                'p6': feats_list[4].cpu(),
            }
            torch.save(single_feats, out_dir / f'{image_id}.pt')

    logging.info(f'Saved reconstructed features to {out_dir}')


if __name__ == '__main__':
    main()


