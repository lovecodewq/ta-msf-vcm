import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def _resolve_binary(preferred: Optional[str], env_var: str, candidates: List[str]) -> str:
    # 1) explicit path/name
    if preferred:
        # absolute path or name in PATH
        if Path(preferred).exists():
            return str(preferred)
        found = shutil.which(preferred)
        if found:
            return found
    # 2) environment variable
    from_env = os.environ.get(env_var)
    if from_env:
        if Path(from_env).exists():
            return from_env
        found = shutil.which(from_env)
        if found:
            return found
    # 3) common candidate names
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(f"Could not resolve binary. Tried preferred='{preferred}', env_var={env_var}, candidates={candidates}")


def run_encoder(
    in_yuv: Path,
    width: int,
    height: int,
    bitdepth: int,
    chroma_format: str,
    qp: int,
    out_bitstream: Path,
    encoder_bin: str = 'EncoderApp',
    encoder_cfg: Optional[List[Path]] = None,
    intra_only: bool = True,
) -> None:
    out_bitstream.parent.mkdir(parents=True, exist_ok=True)
    enc = _resolve_binary(encoder_bin, 'VTM_ENCODER_BIN', ['EncoderApp', 'EncoderAppStatic'])
    cmd = [enc]
    if encoder_cfg:
        for cfg in encoder_cfg:
            cmd += ['-c', str(cfg)]
    cmd += [
        '-i', str(in_yuv),
        '-b', str(out_bitstream),
        '-wdt', str(width),
        '-hgt', str(height),
        '-fr', '1',
        '-f', '1',
        '-q', str(qp),
    ]
    # Do not pass bitdepth/chroma flags to maximize compatibility across VTM builds.
    # Expect the input YUV to match VTM defaults (8-bit, 4:2:0) unless overridden via cfg.
    if intra_only:
        # Use short option -ip which is widely supported
        cmd += ['-ip', '1']
    print(f"[VTM ENC] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_decoder(
    bitstream: Path,
    out_yuv: Path,
    decoder_bin: str = 'DecoderApp',
) -> None:
    out_yuv.parent.mkdir(parents=True, exist_ok=True)
    dec = _resolve_binary(decoder_bin, 'VTM_DECODER_BIN', ['DecoderApp', 'DecoderAppStatic'])
    cmd = [dec, '-b', str(bitstream), '-o', str(out_yuv)]
    print(f"[VTM DEC] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ffmpeg_rgb_to_yuv(img_path: Path, out_yuv: Path, pix_fmt: str = 'yuv420p10le') -> None:
    out_yuv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', str(img_path),
        '-pix_fmt', pix_fmt,
        # ensure even dimensions for 4:2:0 by padding to next even
        '-vf', 'scale=iw:ih,pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-f', 'rawvideo', str(out_yuv),
    ]
    print(f"[FFMPEG RGB->YUV] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ffmpeg_yuv_to_png(in_yuv: Path, width: int, height: int, pix_fmt: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', pix_fmt,
        '-s', f'{width}x{height}',
        '-i', str(in_yuv),
        str(out_png),
    ]
    print(f"[FFMPEG YUV->PNG] cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

