"""Loads the experimental configuration (configs/default.yaml) into CFG."""
import os
import yaml

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")

def load_config(path=_DEFAULT):
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config(os.environ.get("FEDMED_CONFIG", _DEFAULT))
