from pathlib import Path
import sys
import importlib.util
import types


def load_lmsfc_feature_compressor(third_party_root: Path):
    """Dynamically load L-MSFC FeatureCompressor from third-party tree.

    The L-MSFC code expects `from src.utils.stream_helper import *` inside
    its own module, so we create virtual `src` and `src.utils` packages and
    load the helper under that name before importing `model.py`.
    """
    sh_path = third_party_root / 'L-MSFC' / 'src' / 'utils' / 'stream_helper.py'
    model_path = third_party_root / 'L-MSFC' / 'src' / 'model.py'
    if not sh_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"L-MSFC files not found under {third_party_root}/L-MSFC/src")

    # Create virtual packages for import resolution
    if 'src' not in sys.modules:
        sys.modules['src'] = types.ModuleType('src')
    if 'src.utils' not in sys.modules:
        sys.modules['src.utils'] = types.ModuleType('src.utils')

    # Load stream_helper as 'src.utils.stream_helper'
    spec_sh = importlib.util.spec_from_file_location('src.utils.stream_helper', str(sh_path))
    module_sh = importlib.util.module_from_spec(spec_sh)
    assert spec_sh and spec_sh.loader
    spec_sh.loader.exec_module(module_sh)  # type: ignore[attr-defined]
    sys.modules['src.utils.stream_helper'] = module_sh

    # Load model.py into its own module context
    spec_model = importlib.util.spec_from_file_location('lmsfc_model_module', str(model_path))
    module_model = importlib.util.module_from_spec(spec_model)
    assert spec_model and spec_model.loader
    spec_model.loader.exec_module(module_model)  # type: ignore[attr-defined]

    FeatureCompressor = getattr(module_model, 'FeatureCompressor')
    return FeatureCompressor

