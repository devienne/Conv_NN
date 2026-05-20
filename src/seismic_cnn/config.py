"""Central configuration loader.

Usage
-----
from seismic_cnn.config import load_config
cfg = load_config()                                      # uses defaults
cfg = load_config("configs/experiment_convnetquake.yaml")  # explicit experiment
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


def _project_root() -> Path:
    """Walk upward from this file until pyproject.toml is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()


def load_config(experiment_yaml=None, local_yaml=None) -> SimpleNamespace:
    """Load local + experiment YAML configs and return a SimpleNamespace.

    Merge order: defaults → local.yaml → experiment yaml.
    Computed data paths are injected automatically from project_root.
    """
    cfg: dict = {}

    local_path = Path(local_yaml) if local_yaml else PROJECT_ROOT / "configs" / "local.yaml"
    if local_path.exists():
        with open(local_path) as f:
            cfg.update(yaml.safe_load(f) or {})

    exp_path = Path(experiment_yaml) if experiment_yaml else PROJECT_ROOT / "configs" / "experiment_convnetquake.yaml"
    if exp_path.exists():
        with open(exp_path) as f:
            cfg.update(yaml.safe_load(f) or {})

    root = Path(cfg.get("project_root", PROJECT_ROOT))

    # Inject resolved data paths so callers never build paths manually.
    cfg.setdefault("raw_mseed_dir",       str(root / "data" / "raw" / "mseed"))
    cfg.setdefault("raw_catalogs_dir",    str(root / "data" / "raw" / "catalogs"))
    cfg.setdefault("interim_dir",         str(root / "data" / "interim"))
    cfg.setdefault("windows_dir",         str(root / "data" / "processed" / "windows"))
    cfg.setdefault("features_dir",        str(root / "data" / "processed" / "features"))
    cfg.setdefault("models_dir",          str(root / "models"))
    cfg.setdefault("figures_dir",         str(root / "reports" / "figures"))
    cfg.setdefault("comparisons_dir",     str(root / "reports" / "model_comparisons"))

    return SimpleNamespace(**cfg)
