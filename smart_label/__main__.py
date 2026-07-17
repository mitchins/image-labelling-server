"""Compatibility entry point for `python -m smart_label` from this checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main():
    root_main = Path(__file__).resolve().parent.parent / "__main__.py"
    spec = importlib.util.spec_from_file_location("_smart_label_root_main", root_main)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
